import random
from typing import List, Tuple, Dict, Optional, Union

import torch
import torch_geometric.data
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch


class TextGraphDataset(Dataset):
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
                 sentence_max_length: int, concept_max_length: int):

        assert (len(sentence_input_ids) == len(token_ent_binary_masks)
                == len(edge_index_token_idx) == len(edge_index_entity_idx))

        self.bert_tokenizer = tokenizer
        self.sentence_input_ids = sentence_input_ids
        self.token_ent_binary_masks = token_ent_binary_masks
        self.edge_index_token_idx = edge_index_token_idx
        self.edge_index_entity_idx = edge_index_entity_idx

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
        assert masking_mode in TextGraphDataset.MASKING_MODES
        self.masking_mode = masking_mode
        self.neighbors_have_rel = use_rel

        self.MASK_TOKEN_ID: int = self.bert_tokenizer.mask_token_id
        self.CLS_TOKEN_ID: int = self.bert_tokenizer.cls_token_id
        self.SEP_TOKEN_ID: int = self.bert_tokenizer.sep_token_id
        self.PAD_TOKEN_ID: int = self.bert_tokenizer.pad_token_id
        self.follow_batch = None
        self.exclude_keys = None

    def pad_input_ids(self, input_ids: List[List[int]], pad_token, max_length, inp_ids_dtype,
                      return_mask, att_mask_dtype=None) \
            -> Tuple[torch.Tensor, Optional[torch.FloatTensor]]:
        att_masks = None
        if return_mask:
            att_masks = att_mask_dtype(
                [[1, ] * len(lst) + [0, ] * (max_length - len(lst)) for lst in input_ids])
        input_ids = inp_ids_dtype(
            tuple((lst + (pad_token,) * (max_length - len(lst)) for lst in input_ids)))
        return input_ids, att_masks

    def sample_node_neighors_subgraph(self, node_ids_list: List[int], mask_trg_nodes, neighbors_have_rel=False):
        if neighbors_have_rel:
            raise NotImplementedError(f"use_rel : {neighbors_have_rel}")
        num_target_concepts = len(node_ids_list)
        init_node_id = 0
        cum_neigh_sample_size = 0
        edge_trg_index = []
        src_nodes_input_ids: List[int] = []
        trg_nodes_input_ids: List[Tuple[int]] = []

        for trg_node_counter, target_node_id in enumerate(node_ids_list):
            ####################################
            # #### Processing neighbor nodes ###
            ####################################

            # TODO: Что делать, если у ноды нет соседей?
            node_neighbor_ids: Tuple[Tuple[int, int, int]] = self.node_id2adjacency_lists.get(target_node_id, [])
            # TODO: Может, докинуть саму ноду? Но если докину, может случиться лик?

            neigh_sample_size = min(self.max_n_neighbors, len(node_neighbor_ids))
            # TODO: если use_rel = False, то не tuple, а int!!!  У меня это учтено??
            # TODO: Проверить, что здесь точно t[0]
            if neighbors_have_rel:
                neigh_input_ids_list = (random.choice(self.node_id2input_ids[t[0]])
                                        for t in random.sample(node_neighbor_ids, neigh_sample_size))
            else:
                neigh_input_ids_list = (random.choice(self.node_id2input_ids[t])
                                        for t in random.sample(node_neighbor_ids, neigh_sample_size))

            # neigh_input_ids, neigh_att_masks = zip(*neigh_tok_out_list)
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
            # TODO: MASK TARGET CONCEPT OR ZERO TENSOR OPTION!!! IMPORTANT!!!!
            init_node_id += neigh_sample_size
            # TODO: я точно правильно стакаю два набора src?

        edge_src_index = torch.arange(cum_neigh_sample_size)
        graph_data_src_neighbors = torch_geometric.data.Data(x=torch.arange(cum_neigh_sample_size),
                                                             edge_src_index=edge_src_index)

        edge_trg_index = torch.LongTensor(edge_trg_index)
        assert edge_src_index.size() == edge_trg_index.size()

        graph_data_trg_nodes = torch_geometric.data.Data(x=torch.arange(num_target_concepts),
                                                         edge_trg_index=edge_trg_index)
        return src_nodes_input_ids, trg_nodes_input_ids, graph_data_src_neighbors, graph_data_trg_nodes

    def mask_fn(self, input_ids, token_entity_mask, i):
        input_id = input_ids[i]
        m = token_entity_mask[i]

        return input_id if m == 0 else self.MASK_TOKEN_ID

    def __len__(self):
        return len(self.sentence_input_ids)

    def __getitem__(self, idx):
        if self.masking_mode == "text":
            mask_entities, mask_nodes = True, False
        elif self.masking_mode == "graph":
            mask_entities, mask_nodes = False, True
        elif self.masking_mode == "both":
            mask_entities, mask_nodes = True, True
        elif self.masking_mode == "random":
            # TODO: Потом задуматься, нормально ли вообще маскировать both?
            # TODO: Может, какие-то эксперименты этому посвятить?
            mask_entities, mask_nodes = random.choice(TextGraphDataset.TEXT_GRAPH_MASKING_OPTIONS)
        else:
            raise ValueError(f"Invalid masking mode: {self.masking_mode}")

        sentence_input_ids = self.sentence_input_ids[idx]
        token_entity_mask = self.token_ent_binary_masks[idx]
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

        src_nodes_inp_ids, trg_nodes_inp_ids, src_neighbors_graph, trg_nodes_graph = self.sample_node_neighors_subgraph(
            unique_mentioned_concept_ids,
            mask_trg_nodes=mask_nodes,
            neighbors_have_rel=self.neighbors_have_rel)

        batch = {
            "sentence_input_ids": sentence_input_ids,
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
        sent_inp_ids, token_is_entity_mask = [], []
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
            sent_inp_ids.append(sample["sentence_input_ids"])
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
