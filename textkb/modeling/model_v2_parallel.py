import logging
import os

import torch
from torch import nn as nn
from torch.cuda.amp import autocast
from torch_geometric.nn import SimpleConv

from textkb.modeling.heads.link_prediction_heads import LinkScorer
from textkb.modeling.heads.lm_heads import BertLMPredictionHead
from textkb.modeling.modeling_utils import bert_encode, mean_pooling


class ModularAlignmentModel(nn.Module):
    def __init__(self, bert_encoder, node_embs, contrastive_loss, multigpu, mlm_task: bool,
                 contrastive_task: bool, graph_lp_task: bool, intermodal_lp_task: bool, num_rels: int, score_type: str,
                 freeze_node_embs: bool, link_regularizer_weight: float):
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
        # TODO: Recheck everything later
        super(ModularAlignmentModel, self).__init__()
        self.bert_hidden_dim = bert_encoder.config.hidden_size
        self.bert_config = bert_encoder.config
        self.entity_token_aggr_conv = SimpleConv(aggr='mean')
        self.node_embs_matrix = node_embs
        self.freeze_node_embs = freeze_node_embs
        if not isinstance(self.node_embs_matrix, torch.FloatTensor):
            self.node_embs_matrix = torch.FloatTensor(self.node_embs_matrix)
        if freeze_node_embs:
            self.node_embs_matrix.requires_grad = True
        else:
            self.node_embs_matrix.requires_grad = False
        self.mlm_task = mlm_task
        self.contrastive_task = contrastive_task
        self.intermodal_lp_task = intermodal_lp_task
        self.graph_lp_task = graph_lp_task

        self.contrastive_loss = None
        if self.contrastive_task:
            self.contrastive_loss = contrastive_loss
        self.mlm_head = None
        if self.mlm_task:
            self.mlm_head = BertLMPredictionHead(bert_encoder.config)
        self.lp_head = None
        if self.intermodal_lp_task or self.graph_lp_task:
            assert num_rels is not None and score_type is not None
            # TODO: Не надо ли сделать LayerNorm поверх графовых эмбеддингов?
            self.lp_head = LinkScorer(num_rels=num_rels, h_dim=self.bert_hidden_dim, score_type=score_type,
                                      link_regularizer_weight=link_regularizer_weight)
        # TODO: Как мне сделать синхронный параллелизм по данным берта и линкам?
        if multigpu:
            self.bert_encoder = nn.DataParallel(bert_encoder)
        else:
            self.bert_encoder = bert_encoder

    @autocast()
    def forward(self, sentence_input, token_is_entity_mask, entity_node_ids, subtoken2entity_edge_index, num_entities,
                corrupted_sentence_input=None, token_labels=None, pos_triples=None, neg_node_ids=None,
                has_edge_mask=None, link_pred_batch_type=None):
        sent_inp_ids, sent_att_mask = sentence_input

        # <b, seq, h> - embedding sentences
        sent_bert_emb = bert_encode(self.bert_encoder, sent_inp_ids, sent_att_mask)
        # <total_num_entity_tokens, h> - pooling entity tokens from sentences
        bert_entity_token_embs = sent_bert_emb[token_is_entity_mask > 0, :]
        entity_embs = torch.zeros(size=(num_entities, self.bert_hidden_dim), dtype=torch.float32)
        # <num_entities, h> - aggregating tokens into sentence-based entity embeddings
        entity_embs = self.entity_token_aggr_conv(edge_index=subtoken2entity_edge_index,
                                                  x=(bert_entity_token_embs, entity_embs))
        node_embs = self.node_embs_matrix[entity_node_ids]
        if self.mlm_task:
            corr_sent_inp_ids, corr_sent_att_mask = corrupted_sentence_input
            corrupted_sent_bert_emb = bert_encode(self.bert_encoder, corr_sent_inp_ids, corr_sent_att_mask)
            masked_lm_loss = self.mlm_head(corrupted_sent_bert_emb, token_labels)
        else:
            masked_lm_loss = 0.
        assert entity_embs.size() == node_embs.size()

        if self.contrastive_task:
            entity_concept_embs = torch.cat([entity_embs, node_embs], dim=0)
            labels = torch.cat([entity_node_ids, entity_node_ids], dim=0)
            contrastive_loss = self.contrastive_loss(entity_concept_embs, labels, )
        else:
            contrastive_loss = 0.
        # TODO: Сделать проверку на то, что у меня батчи чередуются
        if self.intermodal_lp_task:
            rel_idx = pos_triples[1]
            assert neg_node_ids.dim() == 2

            if link_pred_batch_type == "head":
                # Tail embeddings are textual entity embeddings. Head is corrupted node embeddings
                pos_tail_text_embs = entity_embs[has_edge_mask > 0, :]
                pos_head_idx = pos_triples[0]
                pos_head_node_embs = self.node_embs_matrix[pos_head_idx]
                neg_head_node_embs = self.node_embs_matrix[neg_node_ids]

                pos_score = self.lp_head(pos_head_node_embs, pos_tail_text_embs, rel_idx, mode="single")
                neg_score = self.lp_head(neg_head_node_embs, pos_head_node_embs, rel_idx,
                                         mode=link_pred_batch_type)

            elif link_pred_batch_type == "tail":
                # Head embeddings are textual entity embeddings. Tail is corrupted node embeddings
                pos_head_text_embs = entity_embs[has_edge_mask > 0, :]
                pos_tail_idx = pos_triples[2]
                pos_tail_node_embs = self.node_embs_matrix[pos_tail_idx]
                neg_tail_node_embs = self.node_embs_matrix[neg_node_ids]

                pos_score = self.lp_head(pos_head_text_embs, pos_tail_node_embs, rel_idx, mode="single")
                neg_score = self.lp_head(pos_head_text_embs, neg_tail_node_embs, rel_idx,
                                         mode=link_pred_batch_type)
            else:
                raise ValueError(f"Invalid link_pred_batch_type: {link_pred_batch_type}")
            scores = (pos_score, neg_score)
            intermodal_lp_loss, _, _ = self.lp_head.loss(scores)

        else:
            intermodal_lp_loss = 0.

        if self.graph_lp_task:
            pos_head_idx = pos_triples[0]
            pos_tail_idx = pos_triples[2]
            rel_idx = pos_triples[1]
            assert neg_node_ids.dim() == 2
            pos_head_node_embs = self.node_embs_matrix[pos_head_idx]
            pos_tail_node_embs = self.node_embs_matrix[pos_tail_idx]

            if link_pred_batch_type == "head":
                # Head is corrupted node embeddings
                neg_head_node_embs = self.node_embs_matrix[neg_node_ids]

                pos_score = self.lp_head(pos_head_node_embs, pos_tail_node_embs, rel_idx, mode="single")
                neg_score = self.lp_head(neg_head_node_embs, pos_tail_node_embs, rel_idx,
                                         mode=link_pred_batch_type)
            elif link_pred_batch_type == "tail":
                # Tail is corrupted node embeddings
                neg_tail_node_embs = self.node_embs_matrix[neg_node_ids]

                pos_score = self.lp_head(pos_head_node_embs, pos_tail_node_embs, rel_idx, mode="single")
                neg_score = self.lp_head(pos_head_node_embs, neg_tail_node_embs, rel_idx,
                                         mode=link_pred_batch_type)
            else:
                raise ValueError(f"Invalid link_pred_batch_type: {link_pred_batch_type}")
            scores = (pos_score, neg_score)
            graph_lp_loss, _, _ = self.lp_head.loss(scores)
        else:
            graph_lp_loss = 0.

        return masked_lm_loss, contrastive_loss, graph_lp_loss, intermodal_lp_loss

    def save_model(self, output_dir: str):
        node_embs_matrix_path = os.path.join(output_dir, 'node_embs_matrix.pt')
        torch.save(self.node_embs_matrix, node_embs_matrix_path)

        if self.mlm_head is not None:
            mlm_head_path = os.path.join(output_dir, 'mlm_head.pt')
            torch.save(self.mlm_head.state_dict(), mlm_head_path)
        if self.lp_head is not None:
            lp_head_path = os.path.join(output_dir, 'lp_head.pt')
            torch.save(self.lp_head.state_dict(), lp_head_path)
        self.bert_encoder.save_pretrained(output_dir)

        logging.info("Model saved in {}".format(output_dir))

class AlignmentModel(nn.Module):
    def __init__(self, sentence_bert_encoder, concept_bert_encoder, graph_encoder, contrastive_loss, multigpu,
                 freeze_graph_bert_encoder, freeze_graph_encoder):
        super(AlignmentModel, self).__init__()
        # self.bert_encoder = bert_encoder
        self.bert_hidden_dim = sentence_bert_encoder.config.hidden_size
        assert concept_bert_encoder.config.hidden_size == self.bert_hidden_dim
        self.bert_config = sentence_bert_encoder.config
        self.entity_token_aggr_conv = SimpleConv(aggr='mean')
        self.graph_encoder = graph_encoder
        self.contrastive_loss = contrastive_loss
        self.freeze_graph_bert_encoder = freeze_graph_bert_encoder
        self.freeze_graph_encoder = freeze_graph_encoder

        if multigpu:
            self.sentence_bert_encoder = nn.DataParallel(sentence_bert_encoder)
            self.concept_bert_encoder = nn.DataParallel(concept_bert_encoder)
            # self.graph_encoder = torch_geometric.nn.DataParallel(graph_encoder)
        else:
            self.sentence_bert_encoder = sentence_bert_encoder
            self.concept_bert_encoder = concept_bert_encoder
            # self.graph_encoder = graph_encoder

    @autocast()
    def forward(self, sentence_input, token_is_entity_mask, entity_node_ids, subtoken2entity_edge_index,
                concept_graph_input, concept_graph_edge_index, num_entities):

        sent_inp_ids, sent_att_mask = sentence_input

        # <b, seq, h> - embedding sentences
        sent_bert_emb = bert_encode(self.sentence_bert_encoder, sent_inp_ids, sent_att_mask)
        # <total_num_entity_tokens, h> - pooling entity tokens from sentences
        bert_entity_token_embs = sent_bert_emb[token_is_entity_mask > 0, :]
        entity_embs = torch.zeros(size=(num_entities, self.bert_hidden_dim), dtype=torch.float32)
        # <num_entities, h> - aggregating tokens into sentence-based entity embeddings
        entity_embs = self.entity_token_aggr_conv(edge_index=subtoken2entity_edge_index,
                                                  x=(bert_entity_token_embs, entity_embs))
        (concept_graph_input_ids, concept_graph_att_mask) = concept_graph_input
        # <num_ALL_entities, seq, h> - embedding graph concept names
        if self.freeze_graph_bert_encoder:
            with torch.no_grad():
                # <num_ALL_entities, seq, h> - embedding graph concept names
                graph_concept_embs = bert_encode(self.concept_bert_encoder, concept_graph_input_ids,
                                                 concept_graph_att_mask).detach()
                # <num_ALL_entitizes, h> - mean pooling bert embeddings of concept names
                graph_concept_embs = mean_pooling(graph_concept_embs, concept_graph_att_mask)
        else:
            # <num_ALL_entities, seq, h> - embedding graph concept names
            graph_concept_embs = bert_encode(self.concept_bert_encoder, concept_graph_input_ids,
                                             concept_graph_att_mask)
            # <num_ALL_entities, h> - mean pooling bert embeddings of concept names
            graph_concept_embs = mean_pooling(graph_concept_embs, concept_graph_att_mask)
        # # <num_ALL_entities, h> - mean pooling bert embeddings of concept names
        # graph_concept_embs = self.mean_pooling(graph_concept_embs, concept_graph_att_mask)
        if self.freeze_graph_encoder:
            with torch.no_grad():
                # <num_entities, h> - obtaining graph embeddings for concept names
                graph_concept_embs = self.graph_encoder(x=graph_concept_embs,
                                                        edge_index=concept_graph_edge_index,
                                                        num_trg_nodes=num_entities)[:num_entities].detach()
        else:
            # <num_entities, h> - obtaining graph embeddings for concept names
            graph_concept_embs = self.graph_encoder(x=graph_concept_embs,
                                                    edge_index=concept_graph_edge_index,
                                                    num_trg_nodes=num_entities)[:num_entities]
        assert entity_embs.size() == graph_concept_embs.size()

        entity_concept_embs = torch.cat([entity_embs, graph_concept_embs], dim=0)
        labels = torch.cat([entity_node_ids, entity_node_ids], dim=0)
        contrastive_loss = self.contrastive_loss(entity_concept_embs, labels, )

        return contrastive_loss
