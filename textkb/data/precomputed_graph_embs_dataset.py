import os
import random
from abc import ABC
from typing import List, Tuple, Dict, Union, Iterable

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch

from textkb.data.dataset import AbstractGraphNeighborsDataset, sample_masking_flag, MLMDatasetMixin
from textkb.utils.utils import create_t2hr_adjacency_lists_from_h2rt


class LinkPredictionDatasetMixin:

    @staticmethod
    def create_triple_indices(node_id2adjacency_list):
        hr2t: Dict[Tuple[int, int], List[int]] = {}
        tr2h: Dict[Tuple[int, int], List[int]] = {}

        for trg_node_id, neighbors_list in node_id2adjacency_list.items():
            # if t2rh.get(trg_node_id) is None:
            #     t2rh[trg_node_id] = []
            for (neighbor_id, rel, rela) in neighbors_list:
                if hr2t.get((trg_node_id, rel)) is None:
                    hr2t[(trg_node_id, rel)] = []
                if tr2h.get((neighbor_id, rel)) is None:
                    hr2t[(neighbor_id, rel)] = []
                hr2t[(trg_node_id, rel)].append(neighbor_id)
                tr2h[(neighbor_id, rel)].append(trg_node_id)

        return hr2t, tr2h

    @staticmethod
    def sample_triple(node_id, adj_lists, overall_num_nodes: int, n_neg_samples: int, batch_type: str):
        pos_triple, neg_node_ids = None, None
        if adj_lists.get(node_id) is not None:
            target_concept_neighbors = adj_lists[node_id]
            pos_link = random.choice(target_concept_neighbors)

            pos_t, pos_r = pos_link[0], pos_link[1]
            if batch_type == "tail":
                # Head embeddings are textual entity embeddings. Tail is corrupted node embeddings
                pos_triple = (node_id, pos_r, pos_t)
            elif batch_type == "head":
                # Tail embeddings are textual entity embeddings. Head is corrupted node embeddings
                pos_triple = (pos_t, pos_r, node_id)
            else:
                raise ValueError(f"Invalid batch_type: {batch_type}")

            neg_node_ids = tuple(random.sample(range(overall_num_nodes), n_neg_samples))

        return pos_triple, neg_node_ids


class PrecomputedGraphTextDataset(Dataset, AbstractGraphNeighborsDataset, MLMDatasetMixin, LinkPredictionDatasetMixin):
    LP_BATCH_TYPES = ("head", "tail")

    def __init__(self, tokenizer, sentence_input_ids: List[Tuple[int]], token_ent_binary_masks: List[Tuple[int]],
                 edge_index_token_idx: List[Tuple[int]], edge_index_entity_idx: List[Tuple[int]],  # masking_mode: str,
                 node_id2adjacency_list: Dict[int, Tuple[Union[Tuple[int, int, int], Tuple[int]]]],
                 entity_masking_prob: float, sentence_max_length: int,
                 mlm_probability: float, num_nodes: int, link_negative_sample_size: int,
                 global2local_concept_id: Dict[int, int], corrupt_sentences: bool,
                 token_ent_mask_keep_one_ids_only: bool, debug: bool = False):

        assert (len(sentence_input_ids) == len(token_ent_binary_masks)
                == len(edge_index_token_idx) == len(edge_index_entity_idx))
        self.bert_tokenizer = tokenizer
        self.sentence_input_ids = sentence_input_ids
        self.token_ent_binary_masks = token_ent_binary_masks
        self.edge_index_token_idx = edge_index_token_idx
        self.edge_index_entity_idx = edge_index_entity_idx
        self.h2rt_adjacency_lists = node_id2adjacency_list
        self.t2hr_adjacency_lists = create_t2hr_adjacency_lists_from_h2rt(node_id2adjacency_list)

        self.link_negative_sample_size = link_negative_sample_size
        # hr2t, tr2h = self.create_triple_indices(node_id2adjacency_list)
        # self.hr2t, self.tr2h = hr2t, tr2h
        # self.node_embs_matrix = node_embs_matrix
        self.entity_masking_prob = entity_masking_prob

        self.sentence_max_length = sentence_max_length
        # self.masking_mode = masking_mode
        self.mlm_probability = mlm_probability
        self.num_nodes = num_nodes
        self.nodes_with_a_neighbor = tuple(node_id2adjacency_list.keys())
        self.batch_type_id = 0
        self.global2local_concept_id = global2local_concept_id
        self.corrupt_sentences = corrupt_sentences
        self.token_entity_mask_keep_one_ids = token_ent_mask_keep_one_ids_only
        self.debug = debug

        self.MASK_TOKEN_ID: int = self.bert_tokenizer.mask_token_id
        self.CLS_TOKEN_ID: int = self.bert_tokenizer.cls_token_id
        self.SEP_TOKEN_ID: int = self.bert_tokenizer.sep_token_id
        self.PAD_TOKEN_ID: int = self.bert_tokenizer.pad_token_id
        self.follow_batch = None
        self.exclude_keys = None

    def __len__(self):
        return len(self.sentence_input_ids)

    def sample_graph_triples(self, target_concept_ids: Iterable[int], batch_type):
        # batch_type = random.choice(PrecomputedGraphTextDataset.LP_BATCH_TYPES)
        pos_triples = []
        negative_node_ids = []
        has_edge_mask = []
        for trg_c_id in target_concept_ids:
            if batch_type == "tail":
                # Corrupting tail. Textual entity is a head.
                pos_triple, neg_node_ids = self.sample_triple(node_id=trg_c_id,
                                                              adj_lists=self.h2rt_adjacency_lists,
                                                              overall_num_nodes=self.num_nodes,
                                                              n_neg_samples=self.link_negative_sample_size,
                                                              batch_type=batch_type)

            elif batch_type == "head":
                # Corrupting head. Textual entity is a tail.
                pos_triple, neg_node_ids = self.sample_triple(node_id=trg_c_id,
                                                              adj_lists=self.t2hr_adjacency_lists,
                                                              overall_num_nodes=self.num_nodes,
                                                              n_neg_samples=self.link_negative_sample_size,
                                                              batch_type=batch_type)
            else:
                raise ValueError(f"Invalid batch_type: {batch_type}")

            m = 0 if pos_triple is None else 1

            has_edge_mask.append(m)

            if m == 1:
                pos_triple = (self.global2local_concept_id[pos_triple[0]], pos_triple[1],
                              self.global2local_concept_id[pos_triple[2]])
                pos_triples.append(pos_triple)
                negative_node_ids.append(neg_node_ids)

        return pos_triples, negative_node_ids, has_edge_mask

    def __getitem__(self, idx):
        sentence_input_ids = self.sentence_input_ids[idx]
        token_entity_mask = self.token_ent_binary_masks[idx]

        corr_sentence_input_ids, token_labels = None, None
        if self.corrupt_sentences:
            corr_sentence_input_ids, token_labels = self.mask_tokens_single_sample(sentence_input_ids,
                                                                                   token_entity_mask)

        mask_entities = sample_masking_flag(p_true=self.entity_masking_prob)
        if mask_entities:
            sentence_input_ids = tuple(self.mask_fn(sentence_input_ids, token_entity_mask, i)
                                       for i in range(len(sentence_input_ids)))

        if self.token_entity_mask_keep_one_ids:
            token_entity_mask = tuple(i for i, m in enumerate(token_entity_mask) if m == 1)

        edge_index_token_idx = torch.LongTensor(self.edge_index_token_idx[idx])

        edge_index_entity_idx = self.edge_index_entity_idx[idx]
        unique_mentioned_concept_ids = tuple(set(edge_index_entity_idx))

        concept_id2local_id = {concept_id: i for i, concept_id in enumerate(unique_mentioned_concept_ids)}
        token_node_ids = None
        if self.debug:
            token_node_ids = [self.global2local_concept_id[concept_id] for concept_id in edge_index_entity_idx]

        edge_index_entity_idx = torch.LongTensor([concept_id2local_id[concept_id]
                                                  for concept_id in edge_index_entity_idx])
        # TODO: А ЭТО ТОЧНО ПРАВИЛЬНО?
        # TODO: А global2local_concept_id точно тот же самый, что был задействован в infer_node_embeddings_matrix???
        # TODO: Почему в debug я угнал token_node_ids, а entity_node_ids - идентичный, но его я всё-таки считаю?
        entity_node_ids = [self.global2local_concept_id[i] for i in unique_mentioned_concept_ids]

        sentence_tokens_graph = Data(x=torch.arange(edge_index_token_idx.max() + 1),
                                     token_edge_index=edge_index_token_idx)
        sentence_entities_graph = Data(x=torch.arange(len(unique_mentioned_concept_ids)),
                                       entity_edge_index=edge_index_entity_idx)
        assert edge_index_entity_idx.size() == edge_index_token_idx.size()
        batch_type = self.LP_BATCH_TYPES[self.batch_type_id]
        pos_triples, neg_node_ids, has_edge_mask = self.sample_graph_triples(unique_mentioned_concept_ids,
                                                                             batch_type=batch_type)

        batch = {
            "sentence_input_ids": sentence_input_ids,
            # "corrupted_sentence_input_ids": corr_sentence_input_ids,
            # "token_labels": token_labels,
            "token_entity_mask": token_entity_mask,
            "entity_node_ids": entity_node_ids,
            "sentence_tokens_graph": sentence_tokens_graph,
            "sentence_entities_graph": sentence_entities_graph,
            "pos_triples": pos_triples,
            "neg_node_ids": neg_node_ids,
            "has_edge_mask": has_edge_mask,
            "batch_type": batch_type[0]
        }
        if corr_sentence_input_ids is not None:
            batch["corrupted_sentence_input_ids"] = corr_sentence_input_ids
            batch["token_labels"] = token_labels
        if token_node_ids is not None:
            batch["token_node_ids"] = token_node_ids
        self.edge_index_token_idx[idx] = None
        self.sentence_input_ids[idx] = None
        self.token_ent_binary_masks[idx] = None
        self.edge_index_entity_idx[idx] = None

        return batch

    def collate_fn(self, batch):
        d = {}
        sent_inp_ids, corr_sent_inp_ids, token_is_entity_mask = [], [], []
        token_entity_graph_token_part = []
        token_entity_graph_entity_part = []
        token_labels = []

        batch_num_trg_nodes = 0
        batch_sent_max_length = 0
        batch_num_entities = 0
        # batch_node_embeddings = []
        entity_node_ids = []
        token_node_ids = None
        if self.debug:
            token_node_ids = []

        pos_triples = []
        neg_node_ids = []
        has_edge_mask = []
        batch_types = set()

        for sample in batch:
            entity_node_ids.extend(sample["entity_node_ids"])
            batch_num_entities += len(sample["entity_node_ids"])
            sent_inp_ids.append(sample["sentence_input_ids"])
            if self.corrupt_sentences:
                corr_sent_inp_ids.append(sample["corrupted_sentence_input_ids"])
                token_labels.append(sample["token_labels"])
            batch_sent_max_length = max(batch_sent_max_length, len(sample["sentence_input_ids"]))
            token_is_entity_mask.append(sample["token_entity_mask"])

            token_entity_graph_token_part.append(sample["sentence_tokens_graph"])
            token_entity_graph_entity_part.append(sample["sentence_entities_graph"])
            batch_num_trg_nodes += len(sample["entity_node_ids"])
            if self.debug:
                token_node_ids.extend(sample["token_node_ids"])

            pos_triples.extend(sample["pos_triples"])
            neg_node_ids.extend(sample["neg_node_ids"])
            has_edge_mask.extend(sample["has_edge_mask"])
            batch_types.add(sample["batch_type"])

        assert batch_sent_max_length <= self.sentence_max_length
        assert len(has_edge_mask) == len(entity_node_ids)
        assert len(pos_triples) == len(neg_node_ids)
        sent_inp_ids, sent_att_mask = self.pad_input_ids(input_ids=sent_inp_ids,
                                                         pad_token=self.PAD_TOKEN_ID,
                                                         return_mask=True,
                                                         inp_ids_dtype=torch.LongTensor,
                                                         att_mask_dtype=torch.FloatTensor,
                                                         max_length=batch_sent_max_length)
        if self.corrupt_sentences:
            corr_sent_inp_ids, corr_sent_att_mask = self.pad_input_ids(input_ids=corr_sent_inp_ids,
                                                                       pad_token=self.PAD_TOKEN_ID,
                                                                       return_mask=True,
                                                                       inp_ids_dtype=torch.LongTensor,
                                                                       att_mask_dtype=torch.FloatTensor,
                                                                       max_length=batch_sent_max_length)
            token_labels, _ = self.pad_input_ids(input_ids=token_labels,
                                                 pad_token=self.PAD_TOKEN_ID,
                                                 return_mask=False,
                                                 inp_ids_dtype=torch.LongTensor,
                                                 max_length=batch_sent_max_length)
            corr_sent_input = (corr_sent_inp_ids, corr_sent_att_mask)
            d["corrupted_sentence_input"] = corr_sent_input
            d["token_labels"] = token_labels
        if not self.token_entity_mask_keep_one_ids:
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

        sent_input = (sent_inp_ids, sent_att_mask)

        entity_node_ids = torch.LongTensor(entity_node_ids)
        pos_triples = torch.transpose(torch.LongTensor(pos_triples), 0, 1)

        neg_node_ids = torch.LongTensor(neg_node_ids)
        has_edge_mask = torch.LongTensor(has_edge_mask)
        assert neg_node_ids.dim() == 2

        d["sentence_input"] = sent_input
        d["token_is_entity_mask"] = token_is_entity_mask
        d["entity_node_ids"] = entity_node_ids
        d["subtoken2entity_edge_index"] = subtoken2entity_edge_index
        d["num_entities"] = batch_num_entities
        d["pos_triples"] = pos_triples
        d["neg_node_ids"] = neg_node_ids
        d["has_edge_mask"] = has_edge_mask
        d["batch_type"] = self.LP_BATCH_TYPES[self.batch_type_id]
        d["token_ent_mask_keep_one_ids_only"] = self.token_entity_mask_keep_one_ids

        if self.debug:
            d["token_node_ids"] = token_node_ids
        batch_letter = self.LP_BATCH_TYPES[self.batch_type_id][0]
        assert batch_letter in batch_types and len(batch_types) == 1

        self.batch_type_id = (self.batch_type_id + 1) % 2

        return d


class CachedPrecomputedGraphTextDataset(Dataset, AbstractGraphNeighborsDataset, MLMDatasetMixin):

    def __init__(self, tokenizer, data_dir: str, num_batches: int, entity_masking_prob: float, sentence_max_length: int,
                 mlm_probability: float, num_nodes: int, link_negative_sample_size: int):
        self.bert_tokenizer = tokenizer
        self.data_dir = data_dir
        self.num_batches = num_batches
        # self.sentence_input_ids = sentence_input_ids
        # self.token_ent_binary_masks = token_ent_binary_masks
        # self.edge_index_token_idx = edge_index_token_idx
        # self.edge_index_entity_idx = edge_index_entity_idx
        self.entity_masking_prob = entity_masking_prob

        self.sentence_max_length = sentence_max_length
        # self.masking_mode = masking_mode
        self.mlm_probability = mlm_probability
        self.num_nodes = num_nodes
        self.link_negative_sample_size = link_negative_sample_size
        # TODO: Выкинуть те поля, которые не понадобятся позже
        # TODO: ПОЧЕМУ Я МАСКУ ПОРТИЛ, А НИЧЕГО НЕ ПАДАЛО? дОЛЖНО БЫЛО УПАСТЬ!!! А МОЖЕТ БЫТЬ И НЕ ДОЛЖНО, НО
        # TODO: МНЕ ВСЁ РАВНО НУЖНО ДОБАВИТЬ АССЕРТ

        self.MASK_TOKEN_ID: int = self.bert_tokenizer.mask_token_id
        self.CLS_TOKEN_ID: int = self.bert_tokenizer.cls_token_id
        self.SEP_TOKEN_ID: int = self.bert_tokenizer.sep_token_id
        self.PAD_TOKEN_ID: int = self.bert_tokenizer.pad_token_id

        self.skip_first_n_steps = None
        self.steps_counter = 0
        self.return_nothing = False
        # self.follow_batch = None
        # self.exclude_keys = None

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if self.skip_first_n_steps is not None:
            self.steps_counter += 1
            if self.steps_counter == self.skip_first_n_steps:
                self.steps_counter = 0
                self.skip_first_n_steps = None
                self.return_nothing = None
            # To skip first N steps after checkpoint load, we return empty batch dictionary
            return dict()

        batch_path = os.path.join(self.data_dir, f"batch_{idx}.pt")

        d = torch.load(batch_path)
        sentence_input_ids = d["sentence_input_ids"].to(torch.long)
        sentence_att_mask = d["sentence_att_mask"].to(torch.float32)
        # token_is_entity_mask = d["token_is_entity_mask"]
        # entity_node_ids = d["entity_node_ids"]
        # subtoken2entity_edge_index = d["subtoken2entity_edge_index"]
        # num_entities = d["num_entities"]
        pos_triples = d["pos_triples"]
        # has_edge_mask = d["has_edge_mask"]
        # batch_type = d["batch_type"]
        token_ent_mask_keep_one_ids_only = d["token_ent_mask_keep_one_ids_only"]
        if token_ent_mask_keep_one_ids_only:
            token_ent_mask_one_ids = d["token_is_entity_mask"]
            token_is_entity_mask = torch.zeros(size=sentence_input_ids.size(), dtype=torch.long)
            # print("sentence_input_ids", sentence_input_ids.size())
            for i, one_ids in enumerate(token_ent_mask_one_ids):
                token_is_entity_mask[i, one_ids] = 1
            d["token_is_entity_mask"] = token_is_entity_mask

            # token_is_entity_mask[token_is_entity_mask.size(0), token_ent_mask_one_ids] = 1
            # TODO: А как понять, сколько сущностей всего
        # TODO: ТАК! А почему у меня в итоге с маской всё в порядке? я же её убирал
        num_triples = len(pos_triples[0])
        neg_node_ids = torch.randint(high=self.num_nodes, size=(num_triples, self.link_negative_sample_size))

        # print("sentence_input_ids", sentence_input_ids.size(), type(sentence_input_ids))
        corrupted_sentence_input_ids, token_labels = self.mask_tokens(sentence_input_ids)
        # print("corrupted_sentence_input_ids", corrupted_sentence_input_ids.size(), type(corrupted_sentence_input_ids))
        assert (corrupted_sentence_input_ids.size() == sentence_input_ids.size()
                == token_labels.size() == sentence_att_mask.size())
        # print("token_labels", token_labels.size(), type(token_labels))

        d["neg_node_ids"] = neg_node_ids
        d["corrupted_sentence_input"] = (corrupted_sentence_input_ids, sentence_att_mask)
        d["token_labels"] = token_labels
        d["sentence_input"] = (sentence_input_ids, sentence_att_mask)

        return d
