import random
from abc import ABC
from typing import List, Tuple, Dict, Optional, Union, Any

import torch
import torch_geometric.data
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from transformers import AutoTokenizer, PreTrainedTokenizer


def sample_masking_flag(p_true):
    x = random.random()
    flag = False
    if x <= p_true:
        flag = True
    return flag


class AbstractGraphNeighborsDataset(ABC):
    CLS_TOKEN_ID: int
    MASK_TOKEN_ID: int
    SEP_TOKEN_ID: int
    max_n_neighbors: int
    node_id2adjacency_lists: Dict[int, Tuple[Union[Tuple[int, int], Tuple[int]]]]
    node_id2input_ids: List[Tuple[Tuple[int]]]
    masking_mode: str

    @staticmethod
    def pad_input_ids(input_ids: List[List[int]], pad_token, max_length, inp_ids_dtype,
                      return_mask, att_mask_dtype=None) \
            -> Tuple[torch.Tensor, Optional[torch.FloatTensor]]:
        att_masks = None
        if return_mask:
            att_masks = att_mask_dtype(
                [[1, ] * len(lst[:max_length]) + [0, ] * (max_length - len(lst[:max_length])) for lst in input_ids])
        input_ids = inp_ids_dtype(
            tuple((lst[:max_length] + (pad_token,) * (max_length - len(lst[:max_length])) for lst in input_ids)))
        return input_ids, att_masks

    def sample_node_neighors_subgraph(self, node_ids_list: List[int], mask_trg_nodes, neighbors_have_rel=False):

        num_target_concepts = len(node_ids_list)
        init_node_id = 0
        cum_neigh_sample_size = 0
        edge_trg_index = []
        src_nodes_input_ids: List[int] = []
        trg_nodes_input_ids: List[Tuple[int]] = []
        rel_idx: Optional[List[int]] = None
        if neighbors_have_rel:
            rel_idx: List[int] = []
        for trg_node_counter, target_node_id in enumerate(node_ids_list):
            ####################################
            # #### Processing neighbor nodes ###
            ####################################
            node_neighbor_ids: Tuple[Tuple[int, int]] = self.node_id2adjacency_lists.get(target_node_id, [])
            neigh_sample_size = min(self.max_n_neighbors, len(node_neighbor_ids))

            if neigh_sample_size < 2:
                mask_trg_nodes = False
            node_neighbor_ids = random.sample(node_neighbor_ids, neigh_sample_size)
            if neighbors_have_rel:
                neigh_input_ids_list = (random.choice(self.node_id2input_ids[t[0]]) for t in node_neighbor_ids)
                rel_indices = [t[1] for t in node_neighbor_ids]
                rel_idx.extend(rel_indices)

            else:
                neigh_input_ids_list = (random.choice(self.node_id2input_ids[t]) for t in node_neighbor_ids)

            src_nodes_input_ids.extend(neigh_input_ids_list)
            cum_neigh_sample_size += neigh_sample_size
            ####################################
            # ##### Processing target node #####
            ####################################
            # trg_node_input_ids, trg_node_att_mask = random.choice(self.node_id2tokenizer_output[target_node_id])
            if mask_trg_nodes:
                trg_num_tokens = len(random.choice(self.node_id2input_ids[target_node_id])) - 2
                trg_nodes_input_ids.append(
                    (self.CLS_TOKEN_ID,) + (self.MASK_TOKEN_ID,) * trg_num_tokens + (self.SEP_TOKEN_ID,))
            else:
                trg_nodes_input_ids.append(random.choice(self.node_id2input_ids[target_node_id]))
            edge_trg_index.extend([trg_node_counter, ] * neigh_sample_size)

            init_node_id += neigh_sample_size

        edge_src_index = torch.arange(cum_neigh_sample_size)
        if rel_idx is not None:
            assert len(rel_idx) == len(edge_src_index)
        graph_data_src_neighbors = torch_geometric.data.Data(x=torch.arange(cum_neigh_sample_size),
                                                             edge_src_index=edge_src_index)

        edge_trg_index = torch.LongTensor(edge_trg_index)
        assert edge_src_index.size() == edge_trg_index.size()

        graph_data_trg_nodes = torch_geometric.data.Data(x=torch.arange(num_target_concepts),
                                                         edge_trg_index=edge_trg_index)

        return src_nodes_input_ids, trg_nodes_input_ids, graph_data_src_neighbors, graph_data_trg_nodes, rel_idx

    def mask_fn(self, input_ids, token_entity_mask, i):
        input_id = input_ids[i]
        m = token_entity_mask[i]

        return input_id if m == 0 else self.MASK_TOKEN_ID

    def get_masking_flags(self):
        if self.masking_mode == "text":
            mask_entities, mask_nodes = True, False
        elif self.masking_mode == "graph":
            mask_entities, mask_nodes = False, True
        elif self.masking_mode == "both":
            mask_entities, mask_nodes = True, True
        elif self.masking_mode == "random":

            mask_entities, mask_nodes = random.choice(TextGraphGraphNeighborsDataset.TEXT_GRAPH_MASKING_OPTIONS)
        else:
            raise ValueError(f"Invalid masking mode: {self.masking_mode}")
        return mask_entities, mask_nodes


class MLMDatasetMixin:
    bert_tokenizer: PreTrainedTokenizer
    mlm_probability: float

    def mask_tokens(self, inputs: Any,
                    special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        # <b, s>
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.bert_tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        # indices_replaced = (torch.bernoulli(
        #     torch.full(labels.shape, 0.8)).bool() & masked_indices) | token_is_entity_mask_copy
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.bert_tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    # With non MLM on entity tokens
    """
    
    def mask_tokens(self, inputs: Any, mask_mentions, token_is_entity_mask,
                special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
    
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        
        assert inputs.size() == token_is_entity_mask.size()
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        # <b, s>
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.bert_tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
    
        # token_is_entity_mask_copy = token_is_entity_mask.clone()
        token_is_entity_mask_copy = torch.zeros(size=token_is_entity_mask.size(), dtype=torch.bool)
        for j, (mm, tiem) in enumerate(zip(mask_mentions, token_is_entity_mask)):
            assert isinstance(mm, bool)
            if not mm:
                token_is_entity_mask_copy[j] = False
            else:
                token_is_entity_mask_copy[j] = tiem > 0
    
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool() | token_is_entity_mask_copy
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
    
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        # indices_replaced = (torch.bernoulli(
        #     torch.full(labels.shape, 0.8)).bool() & masked_indices) | token_is_entity_mask_copy
        indices_replaced = (torch.bernoulli(
            torch.full(labels.shape, 1.0)).bool() & masked_indices) | token_is_entity_mask_copy
        inputs[indices_replaced] = self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.mask_token)
    
        # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # random_words = torch.randint(len(self.bert_tokenizer), labels.shape, dtype=torch.long)
        # inputs[indices_random] = random_words[indices_random]
    
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    """

    def mask_tokens_single_sample(self, inputs: Any, token_entity_mask,
                                  special_tokens_mask: Optional[Any] = None, ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if not isinstance(inputs, torch.Tensor):
            inputs = torch.LongTensor(inputs)
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        # <b, s>
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            # special_tokens_mask = [
            #     self.bert_tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
            #     labels.tolist()
            # ]
            special_tokens_mask = self.bert_tokenizer.get_special_tokens_mask(labels,
                                                                              already_has_special_tokens=True)
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        # Entities should not be masked
        token_entity_mask = torch.tensor(token_entity_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(token_entity_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.bert_tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        inputs, labels = tuple(inputs.tolist()), tuple(labels.tolist())

        # labels = labels.squeeze(0)
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class TextGraphGraphNeighborsDataset(Dataset, AbstractGraphNeighborsDataset, MLMDatasetMixin):
    BATCH_ORDER = {
        "SUBTOKEN2ENTITY_GRAPH": -1,
    }
    MASKING_MODES = (
        "text",
        "graph",
        "both",
        "random"
    )
    TEXT_GRAPH_MASKING_OPTIONS = ((False, True), (True, False), (True, True))

    def __init__(self, tokenizer, sentence_input_ids: List[Tuple[int]], token_ent_binary_masks: List[Tuple[int]],
                 edge_index_token_idx: List[Tuple[int]], edge_index_entity_idx: List[Tuple[int]],
                 node_id2adjacency_list: Dict[int, Tuple[Union[Tuple[int, int, int], Tuple[int]]]],
                 node_id2input_ids: List[Tuple[Tuple[int]]], max_n_neighbors: int, use_rel: bool, masking_mode: str,
                 sentence_max_length: int, concept_max_length: int, mlm_probability: float):
        # TODO: А у меня нет проблемы в том, что я маскирую предложения и concept names одними и теми же спец.токенами?
        # TODO: Токены ведь должны быть разными!
        assert (len(sentence_input_ids) == len(token_ent_binary_masks)
                == len(edge_index_token_idx) == len(edge_index_entity_idx))

        self.bert_tokenizer = tokenizer
        self.sentence_input_ids = sentence_input_ids
        self.token_ent_binary_masks = token_ent_binary_masks
        self.edge_index_token_idx = edge_index_token_idx
        self.edge_index_entity_idx = edge_index_entity_idx
        # TODO: ОЧЕНЬ ВАЖНО: mentioned_concept_ids !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # self.validate_entities()
        # TODO: На выходе мне надо:
        # Список спанов сабтокенов сущностей каждого предложения: List[List[span_start, span_end]]
        # Список номеров концептов в словаре
        # Нужен граф в виде списка рёбер
        # TODO: Наверное, текстов слишком много для того, чтобы предварительно токенизировать все
        self.node_id2input_ids = node_id2input_ids
        self.node_id2adjacency_lists = node_id2adjacency_list
        self.sentence_max_length = sentence_max_length
        # TODO: USE concept_max_length ! ! ! ! !
        self.concept_max_length = concept_max_length
        self.max_n_neighbors = max_n_neighbors
        assert masking_mode in TextGraphGraphNeighborsDataset.MASKING_MODES
        self.masking_mode = masking_mode
        self.neighbors_have_rel = use_rel
        self.mlm_probability = mlm_probability

        self.MASK_TOKEN_ID: int = self.bert_tokenizer.mask_token_id
        self.CLS_TOKEN_ID: int = self.bert_tokenizer.cls_token_id
        self.SEP_TOKEN_ID: int = self.bert_tokenizer.sep_token_id
        self.PAD_TOKEN_ID: int = self.bert_tokenizer.pad_token_id
        self.follow_batch = None
        self.exclude_keys = None

    def __len__(self):
        return len(self.sentence_input_ids)

    def __getitem__(self, idx):
        mask_entities, mask_nodes = self.get_masking_flags()

        sentence_input_ids = self.sentence_input_ids[idx]
        token_entity_mask = self.token_ent_binary_masks[idx]

        corr_sentence_input_ids, token_labels = self.mask_tokens(sentence_input_ids, token_entity_mask)

        if mask_entities:
            sentence_input_ids = tuple(self.mask_fn(sentence_input_ids, token_entity_mask, i)
                                       for i in range(len(sentence_input_ids)))
        edge_index_token_idx = torch.LongTensor(self.edge_index_token_idx[idx])

        edge_index_entity_idx = self.edge_index_entity_idx[idx]
        unique_mentioned_concept_ids = tuple(set(edge_index_entity_idx))
        concept_id2local_id = {concept_id: i for i, concept_id in enumerate(unique_mentioned_concept_ids)}
        edge_index_entity_idx = torch.LongTensor([concept_id2local_id[concept_id]
                                                  for concept_id in edge_index_entity_idx])

        sentence_tokens_graph = Data(x=torch.arange(edge_index_token_idx.max()),
                                     token_edge_index=edge_index_token_idx)
        sentence_entities_graph = Data(x=torch.arange(len(unique_mentioned_concept_ids)),
                                       entity_edge_index=edge_index_entity_idx)
        # TODO: typing?
        src_nodes_inp_ids, trg_nodes_inp_ids, src_neighbors_graph, trg_nodes_graph, _ = self.sample_node_neighors_subgraph(
            unique_mentioned_concept_ids,
            mask_trg_nodes=mask_nodes,
            neighbors_have_rel=self.neighbors_have_rel)

        batch = {
            "sentence_input_ids": sentence_input_ids,
            "corrupted_sentence_input_ids": corr_sentence_input_ids,
            "token_labels": token_labels,
            "token_entity_mask": token_entity_mask,
            "entity_node_ids": unique_mentioned_concept_ids,
            "sentence_tokens_graph": sentence_tokens_graph,
            "sentence_entities_graph": sentence_entities_graph,
            "neighbors_graph": src_neighbors_graph,
            "trg_nodes_graph": trg_nodes_graph,
            "src_nodes_input_ids": src_nodes_inp_ids,
            "trg_nodes_input_ids": trg_nodes_inp_ids
        }

        return batch

    def collate_fn(self, batch):
        sent_inp_ids, corr_sent_inp_ids, token_is_entity_mask = [], [], []
        # entity_node_ids = []
        # token_concept_ids = []
        src_node_input_ids, trg_node_input_ids = [], []
        # tok_ent_tok_graph = []
        token_entity_graph_token_part = []
        token_entity_graph_entity_part = []
        neighbors_graph = []
        trg_nodes_graph = []

        batch_num_trg_nodes = 0
        batch_sent_max_length = 0
        batch_node_max_length = 0
        batch_num_entities = 0
        entity_node_ids = []

        for sample in batch:
            entity_node_ids.extend(sample["entity_node_ids"])
            batch_num_entities += len(sample["entity_node_ids"])
            sent_inp_ids.append(sample["corrupted_sentence_input_ids"])
            corr_sent_inp_ids.append(sample["sentence_input_ids"])
            batch_sent_max_length = max(batch_sent_max_length, len(sample["sentence_input_ids"]))
            # sent_att_mask.append(sample["tokenized_sentence"]["attention_mask"])
            # token_concept_ids.append(sample["token_concept_ids"])
            token_is_entity_mask.append(sample["token_entity_mask"])
            # tok_ent_tok_graph.append(sample["tok_ent_tok_graph"])
            neighbors_graph.append(sample["neighbors_graph"])
            trg_nodes_graph.append(sample["trg_nodes_graph"])
            token_entity_graph_token_part.append(sample["sentence_tokens_graph"])
            token_entity_graph_entity_part.append(sample["sentence_entities_graph"])
            # print("GGG", sample["trg_nodes_graph"].x.size()[0])
            batch_num_trg_nodes += sample["trg_nodes_graph"].x.size()[0]
            sample_src_nodes_input_ids = sample["src_nodes_input_ids"]
            sample_trg_nodes_input_ids = sample["trg_nodes_input_ids"]
            src_node_input_ids.extend(sample_src_nodes_input_ids)
            trg_node_input_ids.extend(sample_trg_nodes_input_ids)
            # TODO
            # print("sample_src_nodes_input_ids", sample_src_nodes_input_ids)
            # print("sample_trg_nodes_input_ids", sample_trg_nodes_input_ids)
            # print('--')
            # TODO: Почему у меня sample_src_nodes_input_ids пустые?

            src_nodes_max_length = max((len(t) for t in sample_src_nodes_input_ids)) \
                if len(sample_src_nodes_input_ids) != 0 else 0
            trg_nodes_max_length = max((len(t) for t in sample_trg_nodes_input_ids))
            batch_node_max_length = max(batch_node_max_length, src_nodes_max_length, trg_nodes_max_length)

        assert batch_node_max_length <= self.concept_max_length
        assert batch_sent_max_length <= self.sentence_max_length

        sent_inp_ids, sent_att_mask = self.pad_input_ids(input_ids=sent_inp_ids,
                                                         pad_token=self.PAD_TOKEN_ID,
                                                         return_mask=True,
                                                         inp_ids_dtype=torch.LongTensor,
                                                         att_mask_dtype=torch.FloatTensor,
                                                         max_length=batch_sent_max_length)
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
        # TODO: Почему у меня у меня???
        """
        tok_ent_edge_index_token_idx tensor([ 0,  1,  2,  3,  4,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 18, 19, 20, 21, 22, 23, 24])
        tok_ent_edge_index_entity_idx tensor([0, 1, 1, 1, 0, 3, 3, 3, 4, 4, 4, 4, 4, 2, 2, 4, 4, 4, 4, 4, 6, 6, 6, 5,
        5, 5, 5])
        """
        assert len(tok_ent_edge_index_token_idx) == len(tok_ent_edge_index_entity_idx)
        subtoken2entity_edge_index = torch.stack((tok_ent_edge_index_token_idx, tok_ent_edge_index_entity_idx),
                                                 dim=0)
        # TODO: src, trg graphs.
        src_nodes_graph = Batch.from_data_list(neighbors_graph, self.follow_batch, self.exclude_keys)
        trg_nodes_graph = Batch.from_data_list(trg_nodes_graph, self.follow_batch, self.exclude_keys)
        # print('--')
        # print("neighbors_graph", neighbors_graph)
        # print("trg_nodes_graph", trg_nodes_graph)
        #
        # print("src_nodes_graph", src_nodes_graph)
        # print("trg_nodes_graph", trg_nodes_graph)
        # print("src_nodes_edge_index BEFORE", src_nodes_graph.edge_src_index)
        # print("trg_nodes_edge_index BEFORE", trg_nodes_graph.edge_trg_index)
        src_nodes_edge_index = src_nodes_graph.edge_src_index + batch_num_trg_nodes
        trg_nodes_edge_index = trg_nodes_graph.edge_trg_index
        # # TODO!!!
        # print("src_nodes_edge_index AFTER", src_nodes_edge_index)
        # print("trg_nodes_edge_index AFTER", trg_nodes_edge_index)
        # print('--')
        # TODO! ! !
        # print("batch_num_trg_nodes", batch_num_trg_nodes)
        # print("trg_nodes_graph.x.size()", trg_nodes_graph.x.size())
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

        sent_input = (sent_inp_ids, sent_att_mask)
        # token_is_entity_mask = torch.stack(token_is_entity_mask)
        entity_node_ids = torch.LongTensor(entity_node_ids)

        d = {
            "sentence_input": sent_input,
            "token_is_entity_mask": token_is_entity_mask,
            # "token_concept_ids": token_concept_ids,
            "entity_node_ids": entity_node_ids,
            "subtoken2entity_edge_index": subtoken2entity_edge_index,
            "node_input": node_input,
            "concept_graph_edge_index": concept_graph_edge_index,
            "num_entities": batch_num_entities
        }
        return d


class GraphNeighborsDataset(Dataset, AbstractGraphNeighborsDataset):

    def __init__(self, tokenizer, node_id2adjacency_list: Dict[int, Tuple[Union[Tuple[int, int, int], Tuple[int]]]],
                 node_id2input_ids: List[Tuple[Tuple[int]]], node_input_ids, max_n_neighbors: int,
                 use_rel: bool, masking_mode: str, central_node_idx, concept_max_length: int):
        # TODO: Если соседей нет, то может быть вообще не маскировать?
        self.bert_tokenizer = tokenizer
        self.central_node_idx = central_node_idx
        self.node_id2input_ids = node_id2input_ids
        self.node_input_ids = node_input_ids
        self.node_id2adjacency_lists = node_id2adjacency_list
        self.concept_max_length = concept_max_length
        self.max_n_neighbors = max_n_neighbors
        assert masking_mode in TextGraphGraphNeighborsDataset.MASKING_MODES
        self.masking_mode = masking_mode
        self.neighbors_have_rel = use_rel

        self.MASK_TOKEN_ID: int = self.bert_tokenizer.mask_token_id
        self.CLS_TOKEN_ID: int = self.bert_tokenizer.cls_token_id
        self.SEP_TOKEN_ID: int = self.bert_tokenizer.sep_token_id
        self.PAD_TOKEN_ID: int = self.bert_tokenizer.pad_token_id
        self.follow_batch = None
        self.exclude_keys = None

    def __len__(self):
        return len(self.node_input_ids)

    def __getitem__(self, idx: int):
        mask_entities, mask_nodes = self.get_masking_flags()

        central_node_id = self.central_node_idx[idx]
        # print(central_node_id, len(self.node_input_ids))

        src_nodes_inp_ids, trg_nodes_inp_ids, src_neighbors_graph, trg_nodes_graph, _ = self.sample_node_neighors_subgraph(
            node_ids_list=(central_node_id,),
            mask_trg_nodes=mask_nodes,
            neighbors_have_rel=self.neighbors_have_rel)

        trg_node_inp_ids = self.node_input_ids[idx]
        # tokens = self.bert_tokenizer.convert_ids_to_tokens(trg_node_inp_ids)
        # tokens = "".join((x.strip("#") if x.startswith("#") else f" {x}" for x in tokens))
        # print(idx, central_node_id, tokens)
        # # tokens = self.bert_tokenizer.convert_ids_to_tokens(trg_nodes_inp_ids[0])
        # # tokens = "".join((x.strip("#") if x.startswith("#") else f" {x}" for x in tokens))
        # tokens = [self.bert_tokenizer.convert_ids_to_tokens(n_ids) for n_ids in trg_nodes_inp_ids]
        # tokens = ["".join((x.strip("#") if x.startswith("#") else f" {x}" for x in t)) for t in tokens]
        # print(idx, central_node_id, "||".join(tokens))
        #
        # print('--')

        # TODO: Тут
        batch = {
            "neighbors_graph": src_neighbors_graph,
            "trg_nodes_graph": trg_nodes_graph,
            "src_nodes_input_ids": src_nodes_inp_ids,
            "trg_nodes_input_ids": trg_node_inp_ids,
            "central_node_id": central_node_id
        }

        return batch

    def collate_fn(self, batch):
        src_node_input_ids, trg_node_input_ids = [], []
        neighbors_graph = []
        trg_nodes_graph = []
        batch_num_trg_nodes = 0
        batch_node_max_length = 0
        entity_node_ids = []

        for sample in batch:
            neighbors_graph.append(sample["neighbors_graph"])
            trg_nodes_graph.append(sample["trg_nodes_graph"])
            batch_num_trg_nodes += sample["trg_nodes_graph"].x.size()[0]
            sample_src_nodes_input_ids = sample["src_nodes_input_ids"]
            sample_trg_nodes_input_ids = sample["trg_nodes_input_ids"]

            src_node_input_ids.extend(sample_src_nodes_input_ids)

            trg_node_input_ids.append(sample_trg_nodes_input_ids)
            entity_node_ids.append(sample["central_node_id"])

            src_nodes_max_length = max((len(t) for t in sample_src_nodes_input_ids)) \
                if len(sample_src_nodes_input_ids) != 0 else 0

            trg_nodes_max_length = len(sample_trg_nodes_input_ids)
            batch_node_max_length = max(batch_node_max_length, src_nodes_max_length, trg_nodes_max_length)

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
        entity_node_ids = torch.LongTensor(entity_node_ids)

        d = {
            "entity_node_ids": entity_node_ids,
            "node_input": node_input,
            "concept_graph_edge_index": concept_graph_edge_index,
        }
        return d
