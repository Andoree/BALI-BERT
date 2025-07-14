import logging
import os

import torch
from pytorch_metric_learning import miners, losses
from torch import nn as nn
from torch.cuda.amp import autocast
from torch.nn import Parameter, MSELoss, BCELoss, BCEWithLogitsLoss
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import SimpleConv, GATv2Conv
from transformers.models.bert.modeling_bert import BertSelfAttention, BertSelfOutput

from textkb.modeling.dgi import AbstractDGIModel
from textkb.modeling.graph_encoders import TransformerConvLayer, AttentionSumLayer
from textkb.modeling.heads.link_transformations import TransETransformation, DistMultTransformation, \
    RotatETransformation
from textkb.modeling.heads.lm_heads import BertLMPredictionHead
from textkb.modeling.modeling_utils import bert_encode, bert_pooling, shuffle_neighboring_nodes, \
    bert_pool_specific_layer


class ModularAlignmentModelBiEncoder(nn.Module, AbstractDGIModel):
    def __init__(self, bert_encoder, graph_encoder, bert_tokenizer, multigpu, mlm_task: bool,
                 textual_contrastive_task: bool,
                 text_node_contrastive_task: bool,
                 text_node_contrastive_task_mse: bool,
                 textual_contrastive_loss,
                 text_graph_contrastive_loss,
                 device,
                 sentence_emb_pooling,
                 concept_emb_pooling,
                 mention_concept_name_link_prediction_task: bool = False,
                 dgi_task: bool = False,
                 text_node_class_task: bool = False,
                 text_graph_contrastive_task_central: bool = False,
                 text_graph_contrastive_task_corrupted: bool = False,
                 graph_mlm_task: bool = False,
                 cls_constraint_task: bool = False,
                 mention_concept_name_contrastive_loss=None,
                 concept_encoder_nograd: bool = False,
                 link_transform_type=None,
                 num_relations: int = None,
                 use_miner: bool = False,
                 miner_margin: float = 0.2,
                 type_of_triplets: str = "all",
                 entity_aggregation_conv: str = "mean",
                 token_entity_index_type: str = "edge_index",
                 graph_format: str = "edge_index",
                 sentence_encoder_pooling_layer: int = -1,
                 concept_encoder_pooling_layer: int = -1,
                 tokenizer=None,
                 use_intermodal_alignment_network=False,
                 output_debug_path=None,
                 class_neg_sampling_ratio=1.):
        """
        # Потенциально сделать 2 матрицы: маскированную и не маскированную
        # Модульный датасет под разные головы:
        #  1. MLM-голова
        #  2. Link prediction голова
        #  3. Голова на contrastive
        #  4. Голова на какую-то классификацию (NSP?)
        #  5. А нельзя ли что-то придумать с доп.токенами под графовую модальность? Сначала идёт текст, а потом
        #  какое-то количество токенов, инициированных из графа
        """
        super(ModularAlignmentModelBiEncoder, self).__init__()
        self.bert_hidden_dim = bert_encoder.config.hidden_size
        self.bert_config = bert_encoder.config
        assert entity_aggregation_conv in ("mean", "attention", "gat", "weighted")
        self.entity_aggregation_conv = entity_aggregation_conv
        self.bert_tokenizer = bert_tokenizer
        self.device = device
        self.multigpu = multigpu
        self.graph_encoder = graph_encoder
        assert token_entity_index_type in ("edge_index", "matrix")
        self.token_entity_index_type = token_entity_index_type
        if self.token_entity_index_type == "edge_index":
            if entity_aggregation_conv == "mean":
                self.entity_token_aggr_conv = SimpleConv(aggr='mean').to(device)
            elif entity_aggregation_conv == "gat":
                transformer_conv_heads = 2
                transformer_conv_out_channels = self.bert_hidden_dim // transformer_conv_heads
                self.entity_token_aggr_conv = GATv2Conv(in_channels=self.bert_hidden_dim,
                                                        out_channels=transformer_conv_out_channels,
                                                        heads=transformer_conv_heads,
                                                        dropout=0.0,
                                                        add_self_loops=False,
                                                        edge_dim=None,
                                                        share_weights=True)
            elif entity_aggregation_conv == "weighted":
                self.entity_token_aggr_conv = AttentionSumLayer(in_channels=self.bert_hidden_dim)

            elif entity_aggregation_conv == "attention":
                transformer_conv_heads = 2
                # transformer_conv_out_channels = self.bert_hidden_dim // transformer_conv_heads
                self.entity_token_aggr_conv = TransformerConvLayer(in_channels=self.bert_hidden_dim,
                                                                   heads=transformer_conv_heads,
                                                                   dropout_p=0.0)
            else:
                raise RuntimeError(f"Unsupported entity_token_aggr_conv: {entity_aggregation_conv}")
        elif self.token_entity_index_type == "matrix":
            if entity_aggregation_conv == "attention":
                self.entity_attention = BertSelfAttention(config=bert_encoder.config,
                                                          position_embedding_type=None)
                self.entity_output = BertSelfOutput(config=bert_encoder.config)
            elif entity_aggregation_conv == "mean":
                pass
            else:
                raise RuntimeError(f"Invalid entity_aggregation_conv: {entity_aggregation_conv}")

        self.mlm_task = mlm_task
        self.textual_contrastive_task = textual_contrastive_task
        self.text_graph_contrastive_task = text_node_contrastive_task
        self.text_graph_contrastive_task_mse = text_node_contrastive_task_mse
        self.mention_concept_name_link_prediction_task = mention_concept_name_link_prediction_task
        self.dgi_task = dgi_task
        self.text_node_class_task = text_node_class_task
        self.text_graph_contrastive_task_corrupted = text_graph_contrastive_task_corrupted
        self.text_graph_contrastive_task_central = text_graph_contrastive_task_central
        self.graph_mlm_task = graph_mlm_task
        self.cls_constraint_task = cls_constraint_task
        if self.cls_constraint_task:
            self.mse_constraint = MSELoss()
        self.mlm_head = None
        if self.mlm_task:
            self.mlm_head = BertLMPredictionHead(bert_encoder.config)
        if self.textual_contrastive_task:
            self.textual_contrastive_loss = textual_contrastive_loss
        if self.text_graph_contrastive_task or self.text_graph_contrastive_task_corrupted \
                or self.text_graph_contrastive_task_central:
            self.text_graph_contrastive_loss = text_graph_contrastive_loss
        self.link_transform_type = link_transform_type
        if self.mention_concept_name_link_prediction_task:
            self.link_transform_type = link_transform_type
            assert self.link_transform_type in ("distmult", "transe", "rotate")
            self.link_transformation = None
            if self.link_transform_type == "transe":
                self.link_transformation = TransETransformation(num_rels=num_relations,
                                                                h_dim=self.bert_hidden_dim,
                                                                transe_margin_gamma=1.0,
                                                                dist_ord=2).to(device)
            elif self.link_transform_type == "distmult":
                self.link_transformation = DistMultTransformation(num_rels=num_relations,
                                                                  h_dim=self.bert_hidden_dim).to(device)
            elif self.link_transform_type == "rotate":
                self.link_transformation = RotatETransformation(num_rels=num_relations,
                                                                h_dim=self.bert_hidden_dim).to(device)
                self.rotate_contrastive_loss = losses.NTXentLoss(distance=self.link_transformation.get_score)
            else:
                raise ValueError(f"Invalid link transformation: {self.link_transform_type}")
            self.mention_concept_name_contrastive_loss = mention_concept_name_contrastive_loss
        if self.dgi_task:
            self.dgi_weight = Parameter(torch.Tensor(self.bert_hidden_dim, self.bert_hidden_dim))
            uniform(self.bert_hidden_dim, self.dgi_weight)
        if self.text_node_class_task:
            self.pair_classif_linear = nn.Sequential(
                nn.Linear(self.bert_hidden_dim * 2, self.bert_hidden_dim),
                nn.BatchNorm1d(self.bert_hidden_dim),
                nn.Dropout(0.1),
                nn.Linear(self.bert_hidden_dim, 1),
                # nn.Sigmoid()
            )
            self.bce_loss = BCEWithLogitsLoss()

        self.use_miner = use_miner
        self.miner_margin = miner_margin
        if self.use_miner:
            self.miner = miners.TripletMarginMiner(margin=miner_margin, type_of_triplets=type_of_triplets)
        else:
            self.miner = None

        self.sentence_emb_pooling = sentence_emb_pooling
        assert self.sentence_emb_pooling in ("cls", "mean")
        self.concept_emb_pooling = concept_emb_pooling
        assert self.concept_emb_pooling in ("cls", "mean", "mean1")
        self.concept_encoder_nograd = concept_encoder_nograd
        assert graph_format in ("linear", "edge_index")
        self.graph_format = graph_format
        if multigpu:
            self.bert_encoder = nn.DataParallel(bert_encoder)
        else:
            self.bert_encoder = bert_encoder
        self.use_intermodal_alignment_network = use_intermodal_alignment_network
        if self.use_intermodal_alignment_network:
            self.text2graph_transform = nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim)
            self.graph2text_transform = nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim)
        if self.text_graph_contrastive_task_mse:
            self.mse_loss = MSELoss()
        self.class_neg_sampling_ratio = class_neg_sampling_ratio

        self.tokenizer = tokenizer
        self.sentence_encoder_pooling_layer = sentence_encoder_pooling_layer
        self.concept_encoder_pooling_layer = concept_encoder_pooling_layer
        self.output_debug_path = output_debug_path

    @autocast()
    def forward(self, corrupted_sentence_input, token_is_entity_mask, entity_node_ids, subtoken2entity_edge_index,
                entity_index_input, concept_graph_input, concept_graph_edge_index, num_entities,
                token_labels=None, rel_idx=None, lin_graph_input=None, lin_graph_token_labels=None):
        output_debug_file = None
        if self.output_debug_path is not None:
            output_debug_file = open(self.output_debug_path, 'a+', encoding="utf-8")
        losses_dict = {}
        corr_sent_inp_ids, corr_sent_att_mask = corrupted_sentence_input
        # Embedding sentences - <b, seq, h>
        corrupted_sent_bert_emb_dict = bert_encode(self.bert_encoder, corr_sent_inp_ids, corr_sent_att_mask,
                                                   output_hidden_states=True)
        # Pooling entity tokens from sentences - <total_num_entity_tokens, h>
        corrupted_sent_bert_emb_last_layer = corrupted_sent_bert_emb_dict["last_hidden_state"]
        corrupted_sent_bert_emb = bert_pool_specific_layer(bert_output_dict=corrupted_sent_bert_emb_dict,
                                                           reverse_layer_id=self.sentence_encoder_pooling_layer)
        if self.token_entity_index_type == "edge_index":
            bert_entity_token_embs = corrupted_sent_bert_emb[token_is_entity_mask > 0, :]
            if not self.concept_encoder_nograd:
                entity_embs = bert_entity_token_embs[:num_entities]
                entity_embs -= bert_entity_token_embs[:num_entities]
            else:
                entity_embs = torch.zeros(size=(num_entities, self.bert_hidden_dim), dtype=torch.float32,
                                          device=self.device)
            # Aggregating tokens into sentence-based entity embeddings - <num_entities, h>
            entity_embs = self.entity_token_aggr_conv(edge_index=subtoken2entity_edge_index,
                                                      x=(bert_entity_token_embs, entity_embs))
            if self.cls_constraint_task:
                bert_non_entity_token_embs = corrupted_sent_bert_emb[token_is_entity_mask == 0, :]
                nogr_non_entity_token_embs = bert_non_entity_token_embs.clone().detach()

                mse_constraint_loss = self.mse_constraint(bert_non_entity_token_embs, nogr_non_entity_token_embs)
                losses_dict["MSE-CONSTR"] = mse_constraint_loss
                # num_non_entity_embs = bert_non_entity_token_embs.size(0)

        elif self.token_entity_index_type == "matrix":
            entity_index_matrix, entity_matrix_mask, sentence_index = entity_index_input
            assert entity_index_matrix.dim() == 2
            assert entity_index_matrix.size() == entity_matrix_mask.size() == sentence_index.size()
            assert num_entities == entity_index_matrix.size(0)
            num_entities, max_num_tokens_in_entity = entity_index_matrix.size()
            entity_index_matrix = entity_index_matrix.view(-1)
            sentence_index = sentence_index.view(-1)

            entity_embs = corrupted_sent_bert_emb[sentence_index, entity_index_matrix, :]
            assert entity_embs.size() == (num_entities * max_num_tokens_in_entity, self.bert_hidden_dim)
            entity_embs = entity_embs.view((num_entities, max_num_tokens_in_entity, -1))
            if self.entity_aggregation_conv == "mean":
                entity_embs = bert_pooling(entity_embs,
                                           att_mask=entity_matrix_mask,
                                           pooling_type="mean")
            elif self.entity_aggregation_conv == "attention":
                entity_embs = entity_embs * entity_matrix_mask.unsqueeze(-1)
                entity_output = self.entity_attention(hidden_states=entity_embs)
                assert entity_output[0].size() == (num_entities, max_num_tokens_in_entity, self.bert_hidden_dim)
                entity_embs = self.entity_output(entity_output[0], entity_embs)[:, 0]
                assert entity_embs.size() == (num_entities, self.bert_hidden_dim)

        # entity_embs = bert_entity_token_embs[:num_entities]
        # entity_embs -= bert_entity_token_embs[:num_entities]

        # Embedding graph concept names - <num_ALL_entities, seq, h>
        if self.graph_format == "edge_index":
            (concept_graph_input_ids, concept_graph_att_mask) = concept_graph_input
            if self.concept_encoder_nograd:
                with torch.no_grad():
                    textual_concept_embs_dict = bert_encode(self.bert_encoder, concept_graph_input_ids,
                                                            concept_graph_att_mask, output_hidden_states=True)
                    textual_concept_embs = bert_pool_specific_layer(bert_output_dict=textual_concept_embs_dict,
                                                                    reverse_layer_id=self.concept_encoder_pooling_layer)

                    # Pooling bert embeddings of concept names <num_ALL_entities, h>
                    textual_concept_embs = bert_pooling(textual_concept_embs, att_mask=concept_graph_att_mask,
                                                        pooling_type=self.concept_emb_pooling)
            else:
                textual_concept_embs_dict = bert_encode(self.bert_encoder, concept_graph_input_ids,
                                                        concept_graph_att_mask, output_hidden_states=True)
                textual_concept_embs = bert_pool_specific_layer(bert_output_dict=textual_concept_embs_dict,
                                                                reverse_layer_id=self.concept_encoder_pooling_layer)
                # Pooling bert embeddings of concept names <num_ALL_entities, h>
                textual_concept_embs = bert_pooling(textual_concept_embs, att_mask=concept_graph_att_mask,
                                                    pooling_type=self.concept_emb_pooling)
            graph_concept_embs = self.graph_encoder(x=textual_concept_embs,
                                                    edge_index=concept_graph_edge_index,
                                                    num_trg_nodes=num_entities)[:num_entities]

            assert entity_embs.size() == graph_concept_embs.size()
        elif self.graph_format == "linear":
            lin_graph_input_ids, lin_graph_att_mask = lin_graph_input
            lin_graph_embs_dict = bert_encode(self.bert_encoder, lin_graph_input_ids, lin_graph_att_mask,
                                              output_hidden_states=True)
            lin_graph_embs = bert_pool_specific_layer(bert_output_dict=lin_graph_embs_dict,
                                                      reverse_layer_id=self.concept_encoder_pooling_layer)
            graph_concept_embs = bert_pooling(lin_graph_embs, att_mask=lin_graph_att_mask, pooling_type="cls")
        else:
            raise RuntimeError(f"Invalid graph_format: {self.graph_format}")
        if self.use_intermodal_alignment_network:
            graph_concept_embs = self.text2graph_transform(graph_concept_embs)

        if output_debug_file is not None:
            if self.graph_format == "edge_index":
                (concept_graph_input_ids, concept_graph_att_mask) = concept_graph_input
                node_tokens = [self.tokenizer.convert_ids_to_tokens(t, skip_special_tokens=True) for t in
                               concept_graph_input_ids]
                node_names = ["".join((x.strip("#") if x.startswith("#") else f" {x}" for x in t))
                              for t in node_tokens]

            elif self.graph_format == "linear":
                lin_graph_input_ids, lin_graph_att_mask = lin_graph_input
                node_tokens = [self.tokenizer.convert_ids_to_tokens(t, skip_special_tokens=True) for t in
                               lin_graph_input_ids]
                node_names = ["".join((x.strip("#") if x.startswith("#") else f" {x}" for x in t))
                              for t in node_tokens]
            else:
                raise RuntimeError(f"Invalid graph format: {self.graph_format}")
            if self.token_entity_index_type == "matrix":
                entity_token_input_ids = corr_sent_inp_ids[sentence_index.view(-1), entity_index_matrix.view(-1)]
                entity_token_input_ids = entity_token_input_ids.view((num_entities, max_num_tokens_in_entity))
                entity_tokens = [self.tokenizer.convert_ids_to_tokens(t, skip_special_tokens=True)
                                 for t in entity_token_input_ids]
                entity_names = ["".join((x.strip("#") if x.startswith("#") else f" {x}" for x in t))
                                for t in entity_tokens]

                output_debug_file.write(f"Entities:\n")
                assert len(entity_node_ids) == len(entity_names) == len(node_names[:len(entity_names)])
                for ent_id, e_name, node_name in zip(entity_node_ids, entity_names, node_names[:len(entity_names)]):
                    output_debug_file.write(f"\t{e_name} -- CUI:{ent_id.item()} -- {node_name}\n")
            elif self.token_entity_index_type == "edge_index":
                token_idx = subtoken2entity_edge_index[0]
                entity_idx = subtoken2entity_edge_index[1]
                output_debug_file.write(f"entity_idx {len(entity_idx)} {entity_idx.min()} {entity_idx.max()}\n")
                output_debug_file.write(f"token_idx {len(token_idx)} {token_idx.min()} {token_idx.max()}\n")
                assert entity_idx.max() + 1 == num_entities
                output_debug_file.write("Subtoken2entity edges:\n")
                for t_id, e_id in zip(token_idx, entity_idx):
                    entity_input_ids = corr_sent_inp_ids[token_is_entity_mask > 0]
                    token_str = self.tokenizer.convert_ids_to_tokens(entity_input_ids[t_id].item(),
                                                                     skip_special_tokens=True)
                    node_str = node_names[e_id]
                    local_entity_id = entity_node_ids[e_id]
                    s = f"\t{token_str} --> {node_str} (local_id: {e_id}, {local_entity_id})\n"
                    output_debug_file.write(s)

        if self.mlm_task:
            masked_lm_loss = self.mlm_head(corrupted_sent_bert_emb_last_layer, token_labels)
            losses_dict["MLM"] = masked_lm_loss
        if self.use_intermodal_alignment_network:
            entity_embs = self.text2graph_transform(entity_embs)

        if self.textual_contrastive_task:

            textual_mention_concept_embs = torch.cat([entity_embs, textual_concept_embs[:num_entities]], dim=0)
            labels = torch.cat([entity_node_ids, entity_node_ids], dim=0)
            # tc_loss = self.textual_contrastive_loss(textual_mention_concept_embs, labels, )
            if self.use_miner:
                hard_pairs = self.miner(textual_mention_concept_embs, labels)
                tc_loss = self.textual_contrastive_loss(textual_mention_concept_embs, labels, hard_pairs)
            else:
                tc_loss = self.textual_contrastive_loss(textual_mention_concept_embs, labels)
            losses_dict["T-TCL"] = tc_loss
        assert len(entity_node_ids) == len(entity_embs) == len(graph_concept_embs)
        if self.text_graph_contrastive_task_mse:
            losses_dict["G-MSE"] = self.mse_loss(entity_embs, graph_concept_embs)

        if self.text_graph_contrastive_task or self.text_graph_contrastive_task_corrupted:
            if self.text_graph_contrastive_task_corrupted:
                corr_concept_graph_edge_index = shuffle_neighboring_nodes(concept_graph_edge_index)
                corr_graph_concept_embs = self.graph_encoder(x=textual_concept_embs,
                                                             edge_index=corr_concept_graph_edge_index,
                                                             num_trg_nodes=num_entities)[:num_entities]
                entity_node_ids_corrupted = entity_node_ids.clone()
                entity_node_ids_corrupted += entity_node_ids.max() + 1
                text_graph_concept_embs = torch.cat([entity_embs, graph_concept_embs, corr_graph_concept_embs], dim=0)
                labels = torch.cat([entity_node_ids, entity_node_ids, entity_node_ids_corrupted], dim=0)
                tgc_loss = self.text_graph_contrastive_loss(text_graph_concept_embs, labels)
                losses_dict["CT-GCL"] = tgc_loss
            else:

                text_graph_concept_embs = torch.cat([entity_embs, graph_concept_embs], dim=0)
                labels = torch.cat([entity_node_ids, entity_node_ids], dim=0)
                tgc_loss = self.text_graph_contrastive_loss(text_graph_concept_embs, labels)
                losses_dict["T-GCL"] = tgc_loss
        if self.text_node_class_task:
            num_nodes = len(entity_embs)
            num_neg_pairs = int(num_nodes * self.class_neg_sampling_ratio)
            pos_embs = torch.cat([entity_embs, graph_concept_embs], dim=-1)
            pos_labels = torch.ones(size=(num_nodes,), dtype=torch.float32, device=entity_embs.device)
            assert len(pos_embs.size()) == 2
            assert pos_embs.size()[-1] == self.bert_hidden_dim * 2
            # neg_perm = torch.randperm(num_neg_pairs)
            neg_perm = torch.randint(low=0, high=num_nodes, size=(num_neg_pairs,))
            neg_entity_embs = entity_embs[neg_perm, :]
            neg_entity_labels = entity_node_ids[neg_perm]

            # neg_perm = torch.randperm(num_neg_pairs)
            neg_perm = torch.randint(low=0, high=num_nodes, size=(num_neg_pairs, ))
            neg_graph_concept_embs = graph_concept_embs[neg_perm, :]
            neg_graph_labels = entity_node_ids[neg_perm]

            neg_labels = (neg_entity_labels == neg_graph_labels).type(torch.float32)

            neg_embs = torch.cat([neg_entity_embs, neg_graph_concept_embs], dim=-1)
            assert neg_embs.size() == pos_embs.size()

            class_embs = torch.cat([pos_embs, neg_embs], dim=0)
            class_labels = torch.cat([pos_labels, neg_labels], dim=0)
            probs = self.pair_classif_linear(class_embs).squeeze(-1)
            tgbi_loss = self.bce_loss(probs, class_labels)
            losses_dict["T-GBI"] = tgbi_loss

        if self.mention_concept_name_link_prediction_task:

            concept_graph_src_edge_index = concept_graph_edge_index[0]
            concept_graph_trg_edge_index = concept_graph_edge_index[1]

            head_embs = entity_embs[concept_graph_trg_edge_index, :]
            tail_embs = textual_concept_embs[concept_graph_src_edge_index, :]
            if self.link_transform_type == "rotate":
                self.link_transformation(head_emb=tail_embs, tail_emb=head_embs, rel_idx=rel_idx)
                tail_trans_head_concat_embs = torch.cat([tail_embs, head_embs], dim=0)
                labels = entity_node_ids[concept_graph_trg_edge_index]
                labels = torch.cat([labels, labels], dim=0)

                m_cnc_loss = self.mention_concept_name_contrastive_loss(tail_trans_head_concat_embs, labels)
            else:
                tail_embs_transformed = self.link_transformation.transform(tail_embs, rel_idx)
                tail_trans_head_concat_embs = torch.cat([tail_embs_transformed, head_embs], dim=0)
                labels = entity_node_ids[concept_graph_trg_edge_index]

                # head_embs_transformed = self.link_transformation.transform(head_embs, rel_idx)
                # tail_trans_head_concat_embs = torch.cat([head_embs_transformed, tail_embs], dim=0)
                # labels = entity_node_ids[concept_graph_trg_edge_index]

                labels = torch.cat([labels, labels], dim=0)
                # m_cnc_loss = self.mention_concept_name_contrastive_loss(tail_trans_head_concat_embs, labels)
                if self.use_miner:
                    hard_pairs = self.miner(tail_trans_head_concat_embs, labels)
                    m_cnc_loss = self.mention_concept_name_contrastive_loss(tail_trans_head_concat_embs, labels,
                                                                            hard_pairs)
                else:
                    m_cnc_loss = self.mention_concept_name_contrastive_loss(tail_trans_head_concat_embs, labels)

            losses_dict["M-CN-CL"] = m_cnc_loss
        if self.dgi_task:
            _, corr_cg_edge_index = self.corruption_fn(embs=textual_concept_embs,
                                                       edge_index=concept_graph_edge_index)
            neg_graph_concept_embs = self.graph_encoder(x=textual_concept_embs,
                                                        edge_index=corr_cg_edge_index,
                                                        num_trg_nodes=num_entities)[:num_entities]
            graph_summary = self.summary_fn(graph_concept_embs)

            dgi_loss = self.dgi_loss(graph_concept_embs, neg_graph_concept_embs, graph_summary)
            losses_dict["DGI"] = dgi_loss

        if self.text_graph_contrastive_task_central:
            # central_x = (textual_concept_embs, entity_embs)
            # central_x = torch.zeros(size=textual_concept_embs.size(), dtype=torch.float32,
            # device=entity_embs.device)
            textual_concept_embs_2 = torch.cat((entity_embs, textual_concept_embs[num_entities:]), dim=0)
            # textual_concept_embs_2[:num_entities] = entity_embs
            # central_x[:num_entities] = entity_embs
            # central_x[num_entities:] = textual_concept_embs[num_entities:]
            graph_entity_central_embs = self.graph_encoder(x=textual_concept_embs_2,
                                                           edge_index=concept_graph_edge_index,
                                                           num_trg_nodes=num_entities)[:num_entities]
            text_graph_concept_embs = torch.cat([graph_entity_central_embs, graph_concept_embs], dim=0)
            labels = torch.cat([entity_node_ids, entity_node_ids], dim=0)
            ggcl_loss = self.text_graph_contrastive_loss(text_graph_concept_embs, labels)
            losses_dict["GGCL"] = ggcl_loss
        if self.graph_mlm_task:
            masked_lm_loss = self.mlm_head(lin_graph_embs, lin_graph_token_labels)
            losses_dict["G-MLM"] = masked_lm_loss
        if output_debug_file is not None:
            output_debug_file.close()

        return losses_dict

    def save_model(self, output_dir: str):
        graph_encoder_path = os.path.join(output_dir, 'graph_encoder.pt')
        if self.graph_encoder is not None:
            torch.save(self.graph_encoder.state_dict(), graph_encoder_path)
        try:
            self.bert_encoder.save_pretrained(output_dir)
            self.bert_tokenizer.save_pretrained(output_dir)
            logging.info("Model saved in {}".format(output_dir))
        except AttributeError as e:
            self.bert_encoder.module.save_pretrained(output_dir)
            self.bert_tokenizer.save_pretrained(output_dir)
            logging.info("Model saved (Module) in {}".format(output_dir))

    def load_from_checkpoint(self, checkpoint_dir: str):
        graph_encoder_path = os.path.join(checkpoint_dir, 'graph_encoder.pt')
        if self.graph_encoder is not None:
            self.graph_encoder.load_state_dict(torch.load(graph_encoder_path, map_location=self.device))
