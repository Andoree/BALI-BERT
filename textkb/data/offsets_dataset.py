import os
from collections import OrderedDict
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
                 concept_name_masking_prob: float, mention_masking_prob: float, graph_format, lin_graph_max_length,
                 indices=None, token_entity_index_type: str = "edge_index", rel_id2tokenized_name=None,
                 graph_mlm_task=False, linear_graph_format: str = "v1", bool_filter_token2entity: bool = False):

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
        self.indices = indices
        assert token_entity_index_type in ("matrix", "edge_index")
        self.token_entity_index_type = token_entity_index_type
        self.rel_id2tokenized_name = rel_id2tokenized_name
        assert graph_format in ("linear", "edge_index")
        self.graph_format = graph_format
        self.linear_graph_format = linear_graph_format
        self.lin_graph_max_length = lin_graph_max_length
        self.graph_mlm_task = graph_mlm_task
        self.bool_filter_token2entity = bool_filter_token2entity

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.offsets)

    def create_sentence_token_indices(self, token_is_entity_mask, token_idx, entity_idx):
        # Creating row part for 2D <num_entities, max_entity_token_count>
        # Entity idx is needed here for abrupting only
        entity_tokens = [i for i, m in enumerate(token_is_entity_mask) if m == 1]
        entity_id2tokens_list = {}
        token_idx = [entity_tokens[idx] for idx in token_idx]
        assert len(token_idx) == len(entity_idx)
        entity_ids = []
        for token_id, entity_id in zip(token_idx, entity_idx):

            if entity_id2tokens_list.get(entity_id) is None:
                entity_id2tokens_list[entity_id] = []
            entity_id2tokens_list[entity_id].append(token_id)

        cum_length = sum(len(v) for v in entity_id2tokens_list.values())
        assert cum_length == len(entity_idx)
        for k in entity_id2tokens_list.keys():
            v = sorted(entity_id2tokens_list[k])
            entity_id2tokens_list[k] = v
        token_entity_indices = []
        for entity_id, token_ids_list in entity_id2tokens_list.items():

            current_entity = []
            last_token_id = token_ids_list[0]
            for curr_token_id in token_ids_list:
                if curr_token_id - last_token_id == 1 or curr_token_id - last_token_id == 0:
                    current_entity.append(curr_token_id)
                else:
                    token_entity_indices.append(tuple(current_entity))
                    entity_ids.append(entity_id)
                    current_entity = []
                last_token_id = curr_token_id
            if len(current_entity) > 0:
                token_entity_indices.append(tuple(current_entity))
                entity_ids.append(entity_id)

        assert len(token_entity_indices) == len(entity_ids)

        return token_entity_indices, entity_ids

    def linearize_graph(self, src_nodes_inp_ids, trg_nodes_inp_ids,
                        src_neighbors_graph, trg_nodes_graph, rel_idx, mask_concept_names):
        edge_src_index = src_neighbors_graph.edge_src_index
        edge_trg_index = trg_nodes_graph.edge_trg_index
        assert len(edge_src_index) == len(edge_trg_index) == len(rel_idx)
        linearized_graph_input_ids = []
        # concatenate edges
        for target_node_id in range(len(trg_nodes_inp_ids)):
            # Linearized graph starts with the target concept's name
            if self.linear_graph_format == "v1":
                lin_g_inp_ids = trg_nodes_inp_ids[target_node_id]
            elif self.linear_graph_format == "v2":
                lin_g_inp_ids = (self.CLS_TOKEN_ID,)
            else:
                raise RuntimeError(f"Invalid linear_graph_format: {self.linear_graph_format}")
            for src_id, trg_id, rel_id in zip(edge_src_index, edge_trg_index, rel_idx):
                if trg_id != target_node_id:
                    continue
                tokenized_rel = self.rel_id2tokenized_name[rel_id]

                src_input_ids = src_nodes_inp_ids[src_id]
                trg_input_ids = trg_nodes_inp_ids[trg_id]
                old_src_len, old_trg_len = len(src_input_ids), len(trg_input_ids)
                # Truncating [CLS] and [SEP] tokens
                src_input_ids = src_input_ids[1:len(src_input_ids) - 1]
                trg_input_ids = trg_input_ids[1:len(trg_input_ids) - 1]
                assert old_src_len == len(src_input_ids) + 2
                assert old_trg_len == len(trg_input_ids) + 2
                # lin_g_inp_ids += trg_input_ids + tokenized_rel + src_input_ids + (self.SEP_TOKEN_ID,)
                if mask_concept_names:
                    trg_input_ids = (self.MASK_TOKEN_ID,) * len(trg_input_ids)
                if self.linear_graph_format == "v1":
                    lin_g_inp_ids += src_input_ids + tokenized_rel + trg_input_ids + (self.SEP_TOKEN_ID,)
                elif self.linear_graph_format == "v2":
                    lin_g_inp_ids += src_input_ids + tokenized_rel + (self.SEP_TOKEN_ID,) + trg_input_ids + \
                                     (self.SEP_TOKEN_ID,) + (self.SEP_TOKEN_ID,)

                else:
                    raise RuntimeError(f"Invalid linear_graph_format: {self.linear_graph_format}")
            linearized_graph_input_ids.append(lin_g_inp_ids)
        assert len(linearized_graph_input_ids) == len(trg_nodes_inp_ids)

        return linearized_graph_input_ids

    def filter_token2entity_indices(self, token_idx, entity_idx):
        seen_token_ids = set()
        assert len(token_idx) == len(entity_idx)
        new_token_idx, new_entity_idx = [], []
        for tok_id, e_id in zip(token_idx, entity_idx):
            if tok_id in seen_token_ids:
                continue
            new_token_idx.append(tok_id)
            new_entity_idx.append(e_id)
            seen_token_ids.add(tok_id)
        new_token_idx, new_entity_idx = tuple(new_token_idx), tuple(new_entity_idx)

        return new_token_idx, new_entity_idx

    def __getitem__(self, idx):
        if self.indices is not None:
            idx = self.indices[idx]
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
            edge_index_token_idx = data[token_mask_end + 4:ei_tok_idx_end + 4]
            edge_index_entity_idx = data[ei_tok_idx_end + 4:ei_ent_idx_end + 4]
            assert len(edge_index_token_idx) == len(edge_index_entity_idx)
        if self.bool_filter_token2entity:
            edge_index_token_idx, edge_index_entity_idx = self.filter_token2entity_indices(
                token_idx=edge_index_token_idx,
                entity_idx=edge_index_entity_idx)
            assert len(edge_index_token_idx) == len(edge_index_entity_idx)

        mask_concept_names = sample_masking_flag(p_true=self.concept_name_masking_prob)
        mask_mentions = sample_masking_flag(p_true=self.mention_masking_prob)
        if mask_mentions:
            sentence_input_ids = tuple(self.mask_fn(sentence_input_ids, token_entity_mask, i)
                                       for i in range(len(sentence_input_ids)))
        sentence_tokens_graph = None
        sentence_entities_graph = None
        entity_index_matrix = None
        # unique_mentioned_concept_ids = tuple(set(edge_index_entity_idx))
        unique_mentioned_concept_ids = list(OrderedDict.fromkeys(edge_index_entity_idx))
        if self.token_entity_index_type == "edge_index":
            edge_index_token_idx = torch.LongTensor(edge_index_token_idx)
            concept_id2local_id = {concept_id: i for i, concept_id in enumerate(unique_mentioned_concept_ids)}
            edge_index_entity_idx = torch.LongTensor([concept_id2local_id[concept_id]
                                                      for concept_id in edge_index_entity_idx])
            sentence_tokens_graph = Data(x=torch.arange(edge_index_token_idx.max() + 1),
                                         token_edge_index=edge_index_token_idx)
            sentence_entities_graph = Data(x=torch.arange(len(unique_mentioned_concept_ids)),
                                           entity_edge_index=edge_index_entity_idx)
        elif self.token_entity_index_type == "matrix":

            entity_index_matrix, entity_ids = self.create_sentence_token_indices(token_is_entity_mask=token_entity_mask,
                                                                                 token_idx=edge_index_token_idx,
                                                                                 entity_idx=edge_index_entity_idx)
            unique_mentioned_concept_ids = entity_ids
        # TODO: recheck code
        src_nodes_inp_ids, trg_nodes_inp_ids, src_neighbors_graph, trg_nodes_graph, rel_idx = self.sample_node_neighors_subgraph(
            unique_mentioned_concept_ids,
            mask_trg_nodes=mask_concept_names,
            neighbors_have_rel=self.neighbors_have_rel)
        lin_graphs_input_ids = None
        if self.graph_format == "linear":
            lin_graphs_input_ids = self.linearize_graph(src_nodes_inp_ids=src_nodes_inp_ids,
                                                        trg_nodes_inp_ids=trg_nodes_inp_ids,
                                                        src_neighbors_graph=src_neighbors_graph,
                                                        trg_nodes_graph=trg_nodes_graph,
                                                        rel_idx=rel_idx,
                                                        mask_concept_names=mask_concept_names)

        batch = {
            "sentence_input_ids": sentence_input_ids,
            # "corrupted_sentence_input_ids": corr_sentence_input_ids,
            # "token_labels": token_labels,
            "token_entity_mask": token_entity_mask,
            "entity_node_ids": unique_mentioned_concept_ids,
            "sentence_tokens_graph": sentence_tokens_graph,
            "sentence_entities_graph": sentence_entities_graph,
            "entity_index_matrix": entity_index_matrix,
            # "neighbors_graph": src_neighbors_graph,
            # "trg_nodes_graph": trg_nodes_graph,
            # "src_nodes_input_ids": src_nodes_inp_ids,
            # "trg_nodes_input_ids": trg_nodes_inp_ids
        }
        if lin_graphs_input_ids is not None:
            batch["lin_graphs_input_ids"] = lin_graphs_input_ids
        else:
            batch["neighbors_graph"] = src_neighbors_graph
            batch["trg_nodes_graph"] = trg_nodes_graph
            batch["src_nodes_input_ids"] = src_nodes_inp_ids
            batch["trg_nodes_input_ids"] = trg_nodes_inp_ids

        if rel_idx is not None:
            batch["rel_idx"] = rel_idx

        # TODO: Всё детально сравнить с тем датасетом, в котором я использовал sample_masking_flag:
        # TODO: возможно, я ошибочно скопировал не то: исправил баги, а в скопированном они не исправлены

        return batch

    def collate_fn(self, batch):
        sent_inp_ids = []
        src_node_input_ids, trg_node_input_ids = None, None
        neighbors_graph, trg_nodes_graph = None, None
        batch_num_trg_nodes, batch_node_max_length = None, None
        batch_lin_graphs_input_ids, batch_lin_graph_max_length = None, None
        if self.graph_format == "edge_index":
            src_node_input_ids, trg_node_input_ids = [], []
            neighbors_graph, trg_nodes_graph = [], []
            batch_num_trg_nodes = 0
            batch_node_max_length = 0
        elif self.graph_format == "linear":
            batch_lin_graphs_input_ids = []
        else:
            raise RuntimeError(f"Invalid graph_format: {self.graph_format}")

        token_entity_graph_token_part, token_entity_graph_entity_part = None, None
        subtoken2entity_edge_index = None
        token_is_entity_mask = None
        if self.token_entity_index_type == "edge_index":
            token_is_entity_mask = []
            token_entity_graph_token_part = []
            token_entity_graph_entity_part = []
        batch_entity_index_matrix, batch_sentence_index = None, None
        batch_tokens_in_entity_max = 0
        if self.token_entity_index_type == "matrix":
            batch_entity_index_matrix = []
            batch_sentence_index = []

        # batch_num_trg_nodes = 0
        batch_sent_max_length = 0

        batch_num_entities = 0
        entity_node_ids = []
        batch_rel_idx = None
        if "rel_idx" in batch[0].keys():
            batch_rel_idx = []

        for s_id, sample in enumerate(batch):
            entity_node_ids.extend(sample["entity_node_ids"])
            batch_num_entities += len(sample["entity_node_ids"])
            sent_inp_ids.append(sample["sentence_input_ids"])
            # corr_sent_inp_ids.append(sample["sentence_input_ids"])
            batch_sent_max_length = max(batch_sent_max_length, len(sample["sentence_input_ids"]))

            if self.graph_format == "edge_index":
                neighbors_graph.append(sample["neighbors_graph"])
                trg_nodes_graph.append(sample["trg_nodes_graph"])
                batch_num_trg_nodes += sample["trg_nodes_graph"].x.size()[0]
                sample_src_nodes_input_ids = sample["src_nodes_input_ids"]
                sample_trg_nodes_input_ids = sample["trg_nodes_input_ids"]
                src_node_input_ids.extend(sample_src_nodes_input_ids)
                trg_node_input_ids.extend(sample_trg_nodes_input_ids)

                src_nodes_max_length = max((len(t) for t in sample_src_nodes_input_ids)) \
                    if len(sample_src_nodes_input_ids) != 0 else 0
                trg_nodes_max_length = max((len(t) for t in sample_trg_nodes_input_ids))
                batch_node_max_length = max(batch_node_max_length, src_nodes_max_length, trg_nodes_max_length)
            elif self.graph_format == "linear":
                lin_graphs_input_ids = sample["lin_graphs_input_ids"]
                batch_lin_graphs_input_ids.extend(lin_graphs_input_ids)

            if self.token_entity_index_type == "edge_index":
                token_is_entity_mask.append(sample["token_entity_mask"])
                token_entity_graph_token_part.append(sample["sentence_tokens_graph"])
                token_entity_graph_entity_part.append(sample["sentence_entities_graph"])
            if self.token_entity_index_type == "matrix":
                entity_index_matrix = sample["entity_index_matrix"]
                tokens_in_entity_max = max((len(t) for t in entity_index_matrix)) \
                    if len(entity_index_matrix) != 0 else 0
                batch_tokens_in_entity_max = max(tokens_in_entity_max, batch_tokens_in_entity_max)

                batch_entity_index_matrix.extend(entity_index_matrix)
                batch_sentence_index.extend([(s_id,) * len(t) for t in entity_index_matrix])
                # batch_sentence_index.extend([[s_id, ], ] * len(entity_index_matrix))

            if batch_rel_idx is not None:
                batch_rel_idx.extend(sample["rel_idx"])
        if self.graph_format == "linear":
            batch_lin_graph_max_length = max((len(t) for t in batch_lin_graphs_input_ids))
            batch_lin_graph_max_length = min(batch_lin_graph_max_length, self.lin_graph_max_length)
            batch_lin_graphs_input_ids, lin_graph_att_mask = self.pad_input_ids(input_ids=batch_lin_graphs_input_ids,
                                                                                pad_token=self.PAD_TOKEN_ID,
                                                                                return_mask=True,
                                                                                inp_ids_dtype=torch.LongTensor,
                                                                                att_mask_dtype=torch.FloatTensor,
                                                                                max_length=batch_lin_graph_max_length)
            if self.graph_mlm_task:
                batch_lin_graphs_input_ids, lin_graph_token_labels = self.mask_tokens(batch_lin_graphs_input_ids)
        elif self.graph_format == "edge_index":
            assert batch_node_max_length <= self.concept_max_length
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

        assert batch_sent_max_length <= self.sentence_max_length

        sent_inp_ids, sent_att_mask = self.pad_input_ids(input_ids=sent_inp_ids,
                                                         pad_token=self.PAD_TOKEN_ID,
                                                         return_mask=True,
                                                         inp_ids_dtype=torch.LongTensor,
                                                         att_mask_dtype=torch.FloatTensor,
                                                         max_length=batch_sent_max_length)
        # TODO: token_entity_mask надо совместить с [cls] и [sep]
        corr_sentence_input_ids, token_labels = self.mask_tokens(sent_inp_ids)
        # TODO: corrupted матрицу замаскировать целиком
        # TODO: Может, всё-таки надо special_tokens mask??
        entity_index_input = None
        if self.token_entity_index_type == "edge_index":
            token_is_entity_mask, _ = self.pad_input_ids(input_ids=token_is_entity_mask,
                                                         pad_token=0,
                                                         return_mask=False,
                                                         inp_ids_dtype=torch.LongTensor,
                                                         max_length=batch_sent_max_length)
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
        elif self.token_entity_index_type == "matrix":
            batch_entity_index_matrix, batch_entity_matrix_mask = self.pad_input_ids(
                input_ids=batch_entity_index_matrix,
                pad_token=0,
                return_mask=True,
                inp_ids_dtype=torch.LongTensor,
                att_mask_dtype=torch.FloatTensor,
                max_length=batch_tokens_in_entity_max)
            # print("batch_entity_matrix_mask", batch_entity_matrix_mask)
            batch_sentence_index, _ = self.pad_input_ids(
                input_ids=batch_sentence_index,
                pad_token=0,
                return_mask=False,
                inp_ids_dtype=torch.LongTensor,
                max_length=batch_tokens_in_entity_max)
            entity_index_input = (batch_entity_index_matrix, batch_entity_matrix_mask, batch_sentence_index)

        corrupted_sent_input = (corr_sentence_input_ids, sent_att_mask)
        # token_is_entity_mask = torch.stack(token_is_entity_mask)
        entity_node_ids = torch.LongTensor(entity_node_ids)

        d = {
            "corrupted_sentence_input": corrupted_sent_input,
            "token_is_entity_mask": token_is_entity_mask,
            # "token_concept_ids": token_concept_ids,
            "entity_node_ids": entity_node_ids,
            "subtoken2entity_edge_index": subtoken2entity_edge_index,
            "entity_index_input": entity_index_input,
            # "node_input": node_input,
            # "concept_graph_edge_index": concept_graph_edge_index,
            "token_labels": token_labels,
            "num_entities": batch_num_entities
        }
        if self.graph_format == "linear":
            lin_graph_input = (batch_lin_graphs_input_ids, lin_graph_att_mask)
            d["lin_graph_input"] = lin_graph_input
        elif self.graph_format == "edge_index":
            d["node_input"] = node_input
            d["concept_graph_edge_index"] = concept_graph_edge_index
        else:
            raise RuntimeError(f"Invalid graph_format: {self.graph_format}")
        if self.graph_mlm_task:
            d["lin_graph_token_labels"] = lin_graph_token_labels
        if batch_rel_idx is not None:
            batch_rel_idx = torch.LongTensor(batch_rel_idx)
            d["rel_idx"] = batch_rel_idx

        return d
