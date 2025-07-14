import logging
import os
from argparse import ArgumentParser
from typing import List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from textkb.data.dataset import TextGraphGraphNeighborsDataset
from textkb.utils.io import load_tokenized_sentences_data_v2, load_tokenized_concepts, load_adjacency_lists, \
    create_dir_if_not_exists


def create_batches(data_loader, output_dir: str):
    for i, batch in tqdm(data_loader):
        sentence_input = [t for t in batch["sentence_input"]]
        sentence_input_ids, sentence_att_mask = sentence_input

        corrupted_sentence_input = [t for t in batch["corrupted_sentence_input"]]
        corr_sentence_input_ids, corr_sentence_att_mask = corrupted_sentence_input
        token_labels = batch["token_labels"]

        concept_graph_input = [t for t in batch["node_input"]]
        concept_input_ids, concept_att_mask = concept_graph_input

        entity_node_ids = batch["entity_node_ids"]
        token_is_entity_mask = batch["token_is_entity_mask"]
        subtoken2entity_edge_index = batch["subtoken2entity_edge_index"]
        concept_graph_edge_index = batch["concept_graph_edge_index"]
        num_entities = batch["num_entities"]
        assert not sentence_input_ids.requires_grad

        d = {
            "sentence_input_ids": sentence_input_ids,
            "sentence_att_mask": sentence_att_mask,
            "corrupted_sentence_input_ids": corr_sentence_input_ids,
            "corrupted_sentence_att_mask": corr_sentence_att_mask,
            "token_labels": token_labels,
            "concept_input_ids": concept_input_ids,
            "concept_att_mask": concept_att_mask,
            "token_is_entity_mask": token_is_entity_mask,
            "entity_node_ids": entity_node_ids,
            "subtoken2entity_edge_index": subtoken2entity_edge_index,
            "concept_graph_edge_index": concept_graph_edge_index,
            "num_entities": num_entities
        }
        output_path = os.path.join(output_dir, f"batch_{i}.pt")

        torch.save(d, output_path)


def main(args):
    train_tokenized_sentences_path = args.train_tokenized_sentences_path
    val_tokenized_sentences_path = args.val_tokenized_sentences_path
    bert_encoder_name = args.bert_encoder_name
    tokenized_concepts_path = args.tokenized_concepts_path
    graph_data_dir = args.graph_data_dir
    adjacency_lists_path = os.path.join(graph_data_dir, "adjacency_lists")
    use_rel = args.use_rel
    max_n_neighbors = args.max_n_neighbors
    sentence_max_length = args.sentence_max_length
    concept_max_length = args.concept_max_length
    masking_mode = args.masking_mode

    train_output_dir = args.train_output_dir
    val_output_dir = args.val_output_dir

    create_dir_if_not_exists(train_output_dir)
    create_dir_if_not_exists(val_output_dir)

    sentence_bert_tokenizer = AutoTokenizer.from_pretrained(bert_encoder_name)
    tr_tokenized_data_dict = load_tokenized_sentences_data_v2(tokenized_data_dir=train_tokenized_sentences_path)
    node_id2input_ids: List[Tuple[Tuple[int]]] = load_tokenized_concepts(tok_conc_path=tokenized_concepts_path)

    tr_sent_input_ids: List[Tuple[int]] = tr_tokenized_data_dict["input_ids"]
    tr_token_ent_b_masks: List[Tuple[int]] = tr_tokenized_data_dict["token_entity_mask"]
    tr_edge_index_token_idx: List[Tuple[int]] = tr_tokenized_data_dict["edge_index_token_idx"]
    tr_edge_index_entity_idx: List[Tuple[int]] = tr_tokenized_data_dict["edge_index_entity_idx"]

    node_id2adjacency_list = load_adjacency_lists(adjacency_lists_path, use_rel=use_rel)

    train_dataset = TextGraphGraphNeighborsDataset(tokenizer=sentence_bert_tokenizer,
                                                   sentence_input_ids=tr_sent_input_ids,
                                                   token_ent_binary_masks=tr_token_ent_b_masks,
                                                   edge_index_token_idx=tr_edge_index_token_idx,
                                                   edge_index_entity_idx=tr_edge_index_entity_idx,
                                                   node_id2adjacency_list=node_id2adjacency_list,
                                                   node_id2input_ids=node_id2input_ids,
                                                   max_n_neighbors=max_n_neighbors,
                                                   use_rel=use_rel,
                                                   masking_mode=masking_mode,
                                                   sentence_max_length=sentence_max_length,
                                                   concept_max_length=concept_max_length)

    val_tokenized_data_dict = load_tokenized_sentences_data_v2(tokenized_data_dir=val_tokenized_sentences_path)

    val_sent_input_ids: List[Tuple[int]] = val_tokenized_data_dict["input_ids"]
    val_token_ent_b_masks: List[Tuple[int]] = val_tokenized_data_dict["token_entity_mask"]
    val_edge_index_token_idx: List[Tuple[int]] = val_tokenized_data_dict["edge_index_token_idx"]
    val_edge_index_entity_idx: List[Tuple[int]] = val_tokenized_data_dict["edge_index_entity_idx"]

    val_dataset = TextGraphGraphNeighborsDataset(tokenizer=sentence_bert_tokenizer,
                                                 sentence_input_ids=val_sent_input_ids,
                                                 token_ent_binary_masks=val_token_ent_b_masks,
                                                 edge_index_token_idx=val_edge_index_token_idx,
                                                 edge_index_entity_idx=val_edge_index_entity_idx,
                                                 node_id2adjacency_list=node_id2adjacency_list,
                                                 node_id2input_ids=node_id2input_ids,
                                                 max_n_neighbors=max_n_neighbors,
                                                 use_rel=use_rel,
                                                 masking_mode=masking_mode,
                                                 sentence_max_length=sentence_max_length,
                                                 concept_max_length=concept_max_length)
    # TODO: Ещё раз подумать про маскинг. я ведь не могу маскировать и то, и то?
    # TODO: Сохранить количество батчей?
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.dataloader_num_workers,
                                shuffle=False, collate_fn=val_dataset.collate_fn)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.dataloader_num_workers,
                                  shuffle=True, collate_fn=train_dataset.collate_fn)

    create_batches(train_dataloader, train_output_dir)
    create_batches(val_dataloader, val_output_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()

    parser.add_argument("--graph_data_dir", type=str)
    parser.add_argument("--bert_encoder_name")
    parser.add_argument("--train_tokenized_sentences_path", type=str)
    parser.add_argument("--val_tokenized_sentences_path", type=str, default=None, required=False)
    parser.add_argument("--tokenized_concepts_path", type=str)
    parser.add_argument("--max_n_neighbors", type=int)
    parser.add_argument("--sentence_max_length", type=int)
    parser.add_argument("--use_rel", action="store_true")
    parser.add_argument("--concept_max_length", type=int)
    parser.add_argument("--masking_mode", choices=("text", "graph", "both", "random"), type=str)

    parser.add_argument('--dataloader_num_workers', type=int)
    parser.add_argument("--batch_size", type=int, )

    parser.add_argument("--train_output_dir", type=str, )
    parser.add_argument("--val_output_dir", type=str, )

    args = parser.parse_args()
    main(args)
