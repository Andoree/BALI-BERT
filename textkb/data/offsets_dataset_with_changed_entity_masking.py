import os
from typing import List, Tuple, Dict, Union

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

from textkb.data.dataset import AbstractGraphNeighborsDataset, MLMDatasetMixin, sample_masking_flag


class TextGraphGraphNeighborsOffsetDataset(Dataset, AbstractGraphNeighborsDataset, MLMDatasetMixin):
    def __init__(self, tokenizer, input_data_dir, offsets, offset_lowerbounds, offset_upperbounds, offset_filenames,
                 node_id2adjacency_list: Dict[int, Tuple[Union[Tuple[int, int], Tuple[int]]]],
                 node_id2input_ids: List[Tuple[Tuple[int]]], max_n_neighbors: int, use_rel: bool,
                 sentence_max_length: int, concept_max_length: int, mlm_probability: float,
                 concept_name_masking_prob: float, mention_masking_prob: float):

        self.bert_tokenizer = tokenizer
        self.input_data_dir = input_data_dir
        self.offsets = offsets
        self.offset_lowerbounds = offset_lowerbounds
        self.offset_upperbounds = offset_upperbounds
        self.offset_filenames = offset_filenames
        # Список спанов сабтокенов сущностей каждого предложения: List[List[span_start, span_end]]
        # Список номеров концептов в словаре
        # Нужен граф в виде списка рёбер
        self.node_id2input_ids = node_id2input_ids
        self.node_id2adjacency_lists = node_id2adjacency_list
        self.sentence_max_length = sentence_max_length
        self.concept_max_length = concept_max_length
        self.max_n_neighbors = max_n_neighbors
        self.neighbors_have_rel = use_rel
        self.mlm_probability = mlm_probability
        self.concept_name_masking_prob = concept_name_masking_prob
        self.mention_masking_prob = mention_masking_prob

        self.MASK_TOKEN_ID: int = self.bert_tokenizer.mask_token_id
        self.CLS_TOKEN_ID: int = self.bert_tokenizer.cls_token_id
        self.SEP_TOKEN_ID: int = self.bert_tokenizer.sep_token_id
        self.PAD_TOKEN_ID: int = self.bert_tokenizer.pad_token_id
        self.follow_batch = None
        self.exclude_keys = None

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        offset = self.offsets[idx]
        filename = None
        for j, (lb, ub) in enumerate(zip(self.offset_lowerbounds, self.offset_upperbounds)):
            if lb <= idx < ub:
                filename = self.offset_filenames[j]
                break
        assert filename is not None
        fpath = os.path.join(self.input_data_dir, filename)
        with open(fpath, 'r', encoding="utf-8") as inp_file:
            inp_file.seek(offset)
            line = inp_file.readline()

            data = tuple(map(int, line.strip().split(',')))
            inp_ids_end, token_mask_end, ei_tok_idx_end, ei_ent_idx_end = data[:4]
            sentence_input_ids = data[4:inp_ids_end + 4]
            token_entity_mask = data[inp_ids_end + 4:token_mask_end + 4]
            edge_index_token_idx = torch.LongTensor(data[token_mask_end + 4:ei_tok_idx_end + 4])
            edge_index_entity_idx = data[ei_tok_idx_end + 4:ei_ent_idx_end + 4]
            assert len(edge_index_token_idx) == len(edge_index_entity_idx)
        unique_mentioned_concept_ids = tuple(set(edge_index_entity_idx))
        concept_id2local_id = {concept_id: i for i, concept_id in enumerate(unique_mentioned_concept_ids)}
        edge_index_entity_idx = torch.LongTensor([concept_id2local_id[concept_id]
                                                  for concept_id in edge_index_entity_idx])
        mask_concept_names = sample_masking_flag(p_true=self.concept_name_masking_prob)
        mask_mentions = sample_masking_flag(p_true=self.mention_masking_prob)
        # if mask_mentions:
        # sentence_input_ids = tuple(self.mask_fn(sentence_input_ids, token_entity_mask, i)
        #                            for i in range(len(sentence_input_ids)))
        sentence_tokens_graph = Data(x=torch.arange(edge_index_token_idx.max() + 1),
                                     token_edge_index=edge_index_token_idx)
        sentence_entities_graph = Data(x=torch.arange(len(unique_mentioned_concept_ids)),
                                       entity_edge_index=edge_index_entity_idx)
        # TODO: recheck code
        src_nodes_inp_ids, trg_nodes_inp_ids, src_neighbors_graph, trg_nodes_graph, rel_idx = self.sample_node_neighors_subgraph(
            unique_mentioned_concept_ids,
            mask_trg_nodes=mask_concept_names,
            neighbors_have_rel=self.neighbors_have_rel)

        batch = {
            "sentence_input_ids": sentence_input_ids,
            # "corrupted_sentence_input_ids": corr_sentence_input_ids,
            # "token_labels": token_labels,
            "token_entity_mask": token_entity_mask,
            "mask_mentions": mask_mentions,
            "entity_node_ids": unique_mentioned_concept_ids,
            "sentence_tokens_graph": sentence_tokens_graph,
            "sentence_entities_graph": sentence_entities_graph,
            "neighbors_graph": src_neighbors_graph,
            "trg_nodes_graph": trg_nodes_graph,
            "src_nodes_input_ids": src_nodes_inp_ids,
            "trg_nodes_input_ids": trg_nodes_inp_ids
        }
        if rel_idx is not None:
            batch["rel_idx"] = rel_idx

        # TODO: Corruption должен быть в collate_fn
        # TODO: Всё детально сравнить с тем датасетом, в котором я использовал sample_masking_flag:
        # TODO: возможно, я ошибочно скопировал не то: исправил баги, а в скопированном они не исправлены

        return batch

    def collate_fn(self, batch):
        sent_inp_ids = []
        token_is_entity_mask = []
        src_node_input_ids, trg_node_input_ids = [], []
        token_entity_graph_token_part = []
        token_entity_graph_entity_part = []
        neighbors_graph = []
        trg_nodes_graph = []
        mask_mentions = []

        batch_num_trg_nodes = 0
        batch_sent_max_length = 0
        batch_node_max_length = 0
        batch_num_entities = 0
        entity_node_ids = []
        batch_rel_idx = None
        if "rel_idx" in batch[0].keys():
            batch_rel_idx = []

        for sample in batch:
            entity_node_ids.extend(sample["entity_node_ids"])
            batch_num_entities += len(sample["entity_node_ids"])
            sent_inp_ids.append(sample["sentence_input_ids"])
            # corr_sent_inp_ids.append(sample["sentence_input_ids"])
            batch_sent_max_length = max(batch_sent_max_length, len(sample["sentence_input_ids"]))

            token_is_entity_mask.append(sample["token_entity_mask"])
            neighbors_graph.append(sample["neighbors_graph"])
            trg_nodes_graph.append(sample["trg_nodes_graph"])
            token_entity_graph_token_part.append(sample["sentence_tokens_graph"])
            token_entity_graph_entity_part.append(sample["sentence_entities_graph"])
            mask_mentions.append(sample["mask_mentions"])

            batch_num_trg_nodes += sample["trg_nodes_graph"].x.size()[0]
            sample_src_nodes_input_ids = sample["src_nodes_input_ids"]
            sample_trg_nodes_input_ids = sample["trg_nodes_input_ids"]
            src_node_input_ids.extend(sample_src_nodes_input_ids)
            trg_node_input_ids.extend(sample_trg_nodes_input_ids)

            src_nodes_max_length = max((len(t) for t in sample_src_nodes_input_ids)) \
                if len(sample_src_nodes_input_ids) != 0 else 0
            trg_nodes_max_length = max((len(t) for t in sample_trg_nodes_input_ids))
            batch_node_max_length = max(batch_node_max_length, src_nodes_max_length, trg_nodes_max_length)
            if batch_rel_idx is not None:
                batch_rel_idx.extend(sample["rel_idx"])

        assert batch_node_max_length <= self.concept_max_length
        assert batch_sent_max_length <= self.sentence_max_length

        token_is_entity_mask, _ = self.pad_input_ids(input_ids=token_is_entity_mask,
                                                     pad_token=0,
                                                     return_mask=False,
                                                     inp_ids_dtype=torch.LongTensor,
                                                     max_length=batch_sent_max_length)

        sent_inp_ids, sent_att_mask = self.pad_input_ids(input_ids=sent_inp_ids,
                                                         pad_token=self.PAD_TOKEN_ID,
                                                         return_mask=True,
                                                         inp_ids_dtype=torch.LongTensor,
                                                         att_mask_dtype=torch.FloatTensor,
                                                         max_length=batch_sent_max_length)
        # TODO: token_entity_mask надо совместить с [cls] и [sep]
        corr_sentence_input_ids, token_labels = self.mask_tokens(sent_inp_ids, mask_mentions, token_is_entity_mask)
        # TODO: corrupted матрицу замаскировать целиком
        # TODO: Может, всё-таки надо special_tokens mask??

        token_entity_graph_token_batch = Batch.from_data_list(token_entity_graph_token_part,
                                                              self.follow_batch,
                                                              self.exclude_keys)
        token_entity_graph_entity_batch = Batch.from_data_list(token_entity_graph_entity_part,
                                                               self.follow_batch,
                                                               self.exclude_keys)
        tok_ent_edge_index_token_idx = token_entity_graph_token_batch.token_edge_index
        tok_ent_edge_index_entity_idx = token_entity_graph_entity_batch.entity_edge_index

        assert len(tok_ent_edge_index_token_idx) == len(tok_ent_edge_index_entity_idx)
        subtoken2entity_edge_index = torch.stack((tok_ent_edge_index_token_idx, tok_ent_edge_index_entity_idx),
                                                 dim=0)
        src_nodes_graph = Batch.from_data_list(neighbors_graph, self.follow_batch, self.exclude_keys)
        trg_nodes_graph = Batch.from_data_list(trg_nodes_graph, self.follow_batch, self.exclude_keys)

        src_nodes_edge_index = src_nodes_graph.edge_src_index + batch_num_trg_nodes
        trg_nodes_edge_index = trg_nodes_graph.edge_trg_index

        assert batch_num_trg_nodes == trg_nodes_graph.x.size()[0]
        assert src_nodes_edge_index.dim() == trg_nodes_edge_index.dim() == 1
        concept_graph_edge_index = torch.stack((src_nodes_edge_index, trg_nodes_edge_index), dim=0)

        trg_node_input_ids.extend(src_node_input_ids)
        node_input_ids = trg_node_input_ids

        node_input_ids, node_att_mask = self.pad_input_ids(node_input_ids, pad_token=self.PAD_TOKEN_ID,
                                                           inp_ids_dtype=torch.LongTensor,
                                                           att_mask_dtype=torch.FloatTensor,
                                                           max_length=batch_node_max_length,
                                                           return_mask=True)

        node_input = (node_input_ids, node_att_mask)

        corrupted_sent_input = (corr_sentence_input_ids, sent_att_mask)
        # token_is_entity_mask = torch.stack(token_is_entity_mask)
        entity_node_ids = torch.LongTensor(entity_node_ids)

        d = {
            "corrupted_sentence_input": corrupted_sent_input,
            "token_is_entity_mask": token_is_entity_mask,
            # TODO: А точно contrastive работает правильно? У меня из разных предложений не приходят ложные негативные
            # TODO: примеры?
            # "token_concept_ids": token_concept_ids,
            "entity_node_ids": entity_node_ids,
            "subtoken2entity_edge_index": subtoken2entity_edge_index,
            "node_input": node_input,
            "concept_graph_edge_index": concept_graph_edge_index,
            "token_labels": token_labels,
            "num_entities": batch_num_entities
        }
        if batch_rel_idx is not None:
            batch_rel_idx = torch.LongTensor(batch_rel_idx)
            d["rel_idx"] = batch_rel_idx

        return d
