import logging
import os

import torch
from torch import nn as nn
from torch.cuda.amp import autocast
from torch_geometric.nn import SimpleConv
from transformers import AutoModel, AutoTokenizer

from textkb.modeling.heads.link_prediction_heads import LinkScorer
from textkb.modeling.heads.lm_heads import BertLMPredictionHead
from textkb.modeling.modeling_utils import bert_encode, mean_pooling


class ModularAlignmentModel(nn.Module):
    def __init__(self, bert_encoder, bert_tokenizer, node_embs, contrastive_loss, multigpu, mlm_task: bool,
                 contrastive_task: bool, graph_lp_task: bool, intermodal_lp_task: bool, num_rels: int,
                 score_type: str, freeze_node_embs: bool, link_regularizer_weight: float, device,
                 dropout_p: float = 0.1, embedding_transform="static"):
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
        assert embedding_transform in ("static", "fc")
        super(ModularAlignmentModel, self).__init__()
        self.bert_hidden_dim = bert_encoder.config.hidden_size
        self.bert_config = bert_encoder.config
        self.entity_token_aggr_conv = SimpleConv(aggr='mean')
        # self.node_embs_matrix = node_embs
        self.freeze_node_embs = freeze_node_embs
        self.dropout_p = dropout_p
        # num_graph_emb, graph_emb_dim = node_embs.shape
        if node_embs is not None:
            if not isinstance(node_embs, torch.FloatTensor):
                node_embs = torch.FloatTensor(node_embs)
            self.node_embs_matrix = nn.Embedding.from_pretrained(node_embs, )
            if freeze_node_embs:
                logging.info(f"Using frozen node embeddings....")
                self.node_embs_matrix.requires_grad = True
            else:
                logging.info(f"Using learnable node embeddings....")
                self.node_embs_matrix.requires_grad = False
            if embedding_transform == "fc":
                logging.info(f"Using projection of node embeddings...")
                self.node_embs_matrix = nn.Sequential(
                    self.node_embs_matrix,
                    nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout_p),
                    nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim),
                    nn.LayerNorm([self.bert_hidden_dim, ], eps=1e-12, elementwise_affine=True),
                    nn.Dropout(self.dropout_p),

                )
        # TODO: Если эмбеддинги заморожены, я могу матрицу держать на cpu и просто индексировать
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
        self.bert_tokenizer = bert_tokenizer
        self.device = device
        self.multigpu = multigpu
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
        # TODO: Получить на вход input_ids и attention_mask для concept names
        # TODO: сделать bert_encode, [cls] или [mean] pooling
        # TODO: Получить на вход edge_index, получить графовые эмбеддинги, взять только первые из них
        # <total_num_entity_tokens, h> - pooling entity tokens from sentences
        bert_entity_token_embs = sent_bert_emb[token_is_entity_mask > 0, :]
        entity_embs = torch.zeros(size=(num_entities, self.bert_hidden_dim), dtype=torch.float32)
        # <num_entities, h> - aggregating tokens into sentence-based entity embeddings
        entity_embs = self.entity_token_aggr_conv(edge_index=subtoken2entity_edge_index,
                                                  x=(bert_entity_token_embs, entity_embs))
        # node_embs = self.node_embs_matrix[entity_node_ids]
        node_embs = self.node_embs_matrix(entity_node_ids)
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
            # TODO: Тут сделать развилку статичесик эмбеддинги/динамические эмбеддинги + contrastive
            pos_text_embs = entity_embs[has_edge_mask > 0, :]
            intermodal_lp_loss = self.calculate_intermodal_lp_loss_static_node_embs(pos_triples=pos_triples,
                                                                                    neg_node_ids=neg_node_ids,
                                                                                    batch_type=link_pred_batch_type,
                                                                                    pos_text_embs=pos_text_embs)
        else:
            intermodal_lp_loss = 0.

        if self.graph_lp_task:
            # TODO: Тут сделать развилку статичесик эмбеддинги/динамические эмбеддинги + contrastive
            # TODO: А может и не делать. Возможно с графовыми эмбеддингами мне не нужна графовая задача, а надо
            # TODO: 1. спуленный mention маппить в [cls] соседа по графу
            # TODO: 2. спуленный mention маппить в графовый эмбеддинг (0 и 1-hop)
            graph_lp_loss = self.calculate_graph_lp_loss_static_node_embs(self, pos_triples, neg_node_ids,
                                                                          link_pred_batch_type)
        else:
            graph_lp_loss = 0.

        return masked_lm_loss, contrastive_loss, graph_lp_loss, intermodal_lp_loss

    # TODO: Сделать contrastive голову на link prediction!!!!!
    def calculate_graph_lp_loss_static_node_embs(self, pos_triples, neg_node_ids, batch_type):
        pos_head_idx = pos_triples[0]
        pos_tail_idx = pos_triples[2]
        rel_idx = pos_triples[1]
        assert neg_node_ids.dim() == 2
        # pos_head_node_embs = self.node_embs_matrix[pos_head_idx]
        # pos_tail_node_embs = self.node_embs_matrix[pos_tail_idx]
        pos_head_node_embs = self.node_embs_matrix(pos_head_idx)
        pos_tail_node_embs = self.node_embs_matrix(pos_tail_idx)

        if batch_type == "head":
            # Head is corrupted node embeddings
            # neg_head_node_embs = self.node_embs_matrix[neg_node_ids]
            neg_head_node_embs = self.node_embs_matrix(neg_node_ids)

            pos_score = self.lp_head(pos_head_node_embs, pos_tail_node_embs, rel_idx, mode="single")
            neg_score = self.lp_head(neg_head_node_embs, pos_tail_node_embs, rel_idx,
                                     mode=batch_type)
        elif batch_type == "tail":
            # Tail is corrupted node embeddings
            # neg_tail_node_embs = self.node_embs_matrix[neg_node_ids]
            neg_tail_node_embs = self.node_embs_matrix(neg_node_ids)

            pos_score = self.lp_head(pos_head_node_embs, pos_tail_node_embs, rel_idx, mode="single")
            neg_score = self.lp_head(pos_head_node_embs, neg_tail_node_embs, rel_idx,
                                     mode=batch_type)
        else:
            raise ValueError(f"Invalid link_pred_batch_type: {batch_type}")
        scores = (pos_score, neg_score)
        graph_lp_loss, _, _ = self.lp_head.loss(scores)

        return graph_lp_loss

    def calculate_intermodal_lp_loss_static_node_embs(self, pos_triples, neg_node_ids, batch_type, pos_text_embs):
        rel_idx = pos_triples[1]
        assert neg_node_ids.dim() == 2

        if batch_type == "head":
            # Tail embeddings are textual entity embeddings. Head is corrupted node embeddings
            # pos_text_embs = entity_embs[has_edge_mask > 0, :]
            pos_head_idx = pos_triples[0]
            # pos_head_node_embs = self.node_embs_matrix[pos_head_idx]
            # neg_head_node_embs = self.node_embs_matrix[neg_node_ids]
            pos_head_node_embs = self.node_embs_matrix(pos_head_idx)
            neg_head_node_embs = self.node_embs_matrix(neg_node_ids)

            pos_score = self.lp_head(pos_head_node_embs, pos_text_embs, rel_idx, mode="single")
            neg_score = self.lp_head(neg_head_node_embs, pos_head_node_embs, rel_idx,
                                     mode=batch_type)

        elif batch_type == "tail":
            # Head embeddings are textual entity embeddings. Tail is corrupted node embeddings
            # pos_text_embs = entity_embs[has_edge_mask > 0, :]
            pos_tail_idx = pos_triples[2]
            # pos_tail_node_embs = self.node_embs_matrix[pos_tail_idx]
            # neg_tail_node_embs = self.node_embs_matrix[neg_node_ids]
            pos_tail_node_embs = self.node_embs_matrix(pos_tail_idx)
            neg_tail_node_embs = self.node_embs_matrix(neg_node_ids)

            pos_score = self.lp_head(pos_text_embs, pos_tail_node_embs, rel_idx, mode="single")
            neg_score = self.lp_head(pos_text_embs, neg_tail_node_embs, rel_idx,
                                     mode=batch_type)
        else:
            raise ValueError(f"Invalid link_pred_batch_type: {batch_type}")
        scores = (pos_score, neg_score)
        intermodal_lp_loss, _, _ = self.lp_head.loss(scores)

        return intermodal_lp_loss

    def save_model(self, output_dir: str):

        node_embs_matrix_path = os.path.join(output_dir, 'node_embs_matrix.pt')
        torch.save(self.node_embs_matrix.state_dict(), node_embs_matrix_path)
        # torch.save(self.node_embs_matrix, node_embs_matrix_path)

        if self.mlm_head is not None:
            mlm_head_path = os.path.join(output_dir, 'mlm_head.pt')
            torch.save(self.mlm_head.state_dict(), mlm_head_path)
        if self.lp_head is not None:
            lp_head_path = os.path.join(output_dir, 'lp_head.pt')
            torch.save(self.lp_head.state_dict(), lp_head_path)
        try:
            self.bert_encoder.save_pretrained(output_dir)
            self.bert_tokenizer.save_pretrained(output_dir)
            logging.info("Model saved in {}".format(output_dir))
        except AttributeError as e:
            self.bert_encoder.module.save_pretrained(output_dir)
            self.bert_tokenizer.save_pretrained(output_dir)
            logging.info("Model saved (DataParallel) in {}".format(output_dir))

    def load_from_checkpoint(self, checkpoint_dir: str):
        node_embs_matrix_path = os.path.join(checkpoint_dir, 'node_embs_matrix.pt')
        self.node_embs_matrix.load_state_dict(torch.load(node_embs_matrix_path, map_location=self.device))
        logging.info("Node embeddings loaded.")
        # self.node_embs_matrix = torch.load(node_embs_matrix_path, map_location=self.device)
        # MLM head
        if self.mlm_head is not None:
            mlm_head_path = os.path.join(checkpoint_dir, 'mlm_head.pt')
            self.mlm_head.load_state_dict(torch.load(mlm_head_path, map_location=self.device))
            logging.info("MLM head loaded.")
        # Link Prediction head
        if self.lp_head is not None:
            lp_head_path = os.path.join(checkpoint_dir, 'lp_head.pt')
            self.lp_head.load_state_dict(torch.load(lp_head_path, map_location=self.device))
            logging.info("Link prediction head loaded")
        # TODO: deleted!
        # self.bert_encoder = AutoModel.from_pretrained(checkpoint_dir)
        # self.bert_tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

        # torch.save(self.node_embs_matrix, node_embs_matrix_path)
        #
        # if self.mlm_head is not None:
        #     mlm_head_path = os.path.join(output_dir, 'mlm_head.pt')
        #     torch.save(self.mlm_head.state_dict(), mlm_head_path)
        # if self.lp_head is not None:
        #     lp_head_path = os.path.join(output_dir, 'lp_head.pt')
        #     torch.save(self.lp_head.state_dict(), lp_head_path)
        # try:
        #     self.bert_encoder.save_pretrained(output_dir)
        #     self.bert_tokenizer.save_pretrained(output_dir)
        #     logging.info("Model saved in {}".format(output_dir))
        # except AttributeError as e:
        #     self.bert_encoder.module.save_pretrained(output_dir)
        #     self.bert_tokenizer.save_pretrained(output_dir)
        #     logging.info("Model saved (DataParallel) in {}".format(output_dir))
        #


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
