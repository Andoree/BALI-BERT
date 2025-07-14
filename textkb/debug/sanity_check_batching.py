import logging
import os
from argparse import ArgumentParser
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from textkb.data.precomputed_graph_embs_dataset import PrecomputedGraphTextDataset
from textkb.utils.io import load_tokenized_sentences_data_v2, load_adjacency_lists, \
    create_dir_if_not_exists, load_list_elem_per_line
from textkb.utils.umls2graph import filter_adjacency_lists


def create_batches(data_loader, output_dir: str):
    for i, batch in enumerate(tqdm(data_loader)):
        sentence_input = [t for t in batch["sentence_input"]]
        sentence_input_ids, sentence_att_mask = sentence_input

        corr_sentence_input_ids, corr_sentence_att_mask, token_labels = None, None, None
        if batch.get("corrupted_sentence_input") is not None:
            corrupted_sentence_input = [t for t in batch["corrupted_sentence_input"]]
            corr_sentence_input_ids, corr_sentence_att_mask = corrupted_sentence_input
            token_labels = batch["token_labels"]

        # concept_graph_input = [t for t in batch["node_input"]]
        # concept_input_ids, concept_att_mask = concept_graph_input

        entity_node_ids = batch["entity_node_ids"]
        token_is_entity_mask = batch["token_is_entity_mask"]
        subtoken2entity_edge_index = batch["subtoken2entity_edge_index"]
        # concept_graph_edge_index = batch["concept_graph_edge_index"]
        num_entities = batch["num_entities"]
        assert not sentence_input_ids.requires_grad

        pos_triples = batch["pos_triples"]
        neg_node_ids = batch["neg_node_ids"]
        has_edge_mask = batch["has_edge_mask"]
        batch_type = batch["batch_type"]
        token_ent_mask_keep_one_ids_only = batch["token_ent_mask_keep_one_ids_only"]
        assert neg_node_ids.dim() == 2
        # TODO: SANITY CHECK



        d = {
            "sentence_input_ids": sentence_input_ids.to(torch.int32),
            "sentence_att_mask": sentence_att_mask.to(torch.int8),
            # "corrupted_sentence_input_ids": corr_sentence_input_ids,
            # "corrupted_sentence_att_mask": corr_sentence_att_mask,
            # "token_labels": token_labels,
            "token_is_entity_mask": token_is_entity_mask,
            "entity_node_ids": entity_node_ids,
            "subtoken2entity_edge_index": subtoken2entity_edge_index,
            "num_entities": num_entities,
            "pos_triples": pos_triples,
            # "neg_node_ids": neg_node_ids,
            "has_edge_mask": has_edge_mask,
            "batch_type": batch_type,
            "token_ent_mask_keep_one_ids_only": token_ent_mask_keep_one_ids_only
        }
        if corr_sentence_input_ids is not None:
            d["corrupted_sentence_input_ids"] = corr_sentence_input_ids
            d["corrupted_sentence_att_mask"] = corr_sentence_att_mask
            d["token_labels"] = token_labels

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
    sentence_max_length = args.sentence_max_length
    concept_max_length = args.concept_max_length
    # masking_mode = args.masking_mode

    mlm_probability = args.mlm_probability
    entity_masking_probability = args.entity_masking_probability
    link_negative_sample_size = args.link_negative_sample_size

    train_output_dir = args.train_output_dir
    val_output_dir = args.val_output_dir

    create_dir_if_not_exists(train_output_dir)
    create_dir_if_not_exists(val_output_dir)

    sentence_bert_tokenizer = AutoTokenizer.from_pretrained(bert_encoder_name)
    tr_tokenized_data_dict = load_tokenized_sentences_data_v2(tokenized_data_dir=train_tokenized_sentences_path)
    # node_id2input_ids: List[Tuple[Tuple[int]]] = load_tokenized_concepts(tok_conc_path=tokenized_concepts_path)

    tr_sent_input_ids: List[Tuple[int]] = tr_tokenized_data_dict["input_ids"]
    tr_token_ent_b_masks: List[Tuple[int]] = tr_tokenized_data_dict["token_entity_mask"]
    tr_edge_index_token_idx: List[Tuple[int]] = tr_tokenized_data_dict["edge_index_token_idx"]
    tr_edge_index_entity_idx: List[Tuple[int]] = tr_tokenized_data_dict["edge_index_entity_idx"]

    node_id2adjacency_list = load_adjacency_lists(adjacency_lists_path, use_rel=use_rel)

    mentioned_concepts_idx_path = os.path.join(graph_data_dir, "mentioned_concepts_idx")
    mentioned_concept_ids = load_list_elem_per_line(input_path=mentioned_concepts_idx_path, dtype=int)
    global2local_concept_id = {global_id: local_id for local_id, global_id in enumerate(mentioned_concept_ids)}
    num_nodes = len(mentioned_concept_ids)
    node_id2adjacency_list = filter_adjacency_lists(node_id2adjacency_list=node_id2adjacency_list,
                                                    global2local_concept_id=global2local_concept_id,
                                                    ensure_src_in_index=True)

    train_dataset = PrecomputedGraphTextDataset(tokenizer=sentence_bert_tokenizer,
                                                sentence_input_ids=tr_sent_input_ids,
                                                token_ent_binary_masks=tr_token_ent_b_masks,
                                                edge_index_token_idx=tr_edge_index_token_idx,
                                                edge_index_entity_idx=tr_edge_index_entity_idx,
                                                node_id2adjacency_list=node_id2adjacency_list,
                                                # masking_mode=masking_mode,
                                                sentence_max_length=sentence_max_length,
                                                concept_max_length=concept_max_length,
                                                entity_masking_prob=entity_masking_probability,
                                                link_negative_sample_size=link_negative_sample_size,
                                                mlm_probability=mlm_probability,
                                                global2local_concept_id=global2local_concept_id,
                                                num_nodes=num_nodes,
                                                corrupt_sentences=args.corrupt_sentences,
                                                token_ent_mask_keep_one_ids_only=args.token_ent_mask_keep_one_ids_only)

    val_tokenized_data_dict = load_tokenized_sentences_data_v2(tokenized_data_dir=val_tokenized_sentences_path)

    val_sent_input_ids: List[Tuple[int]] = val_tokenized_data_dict["input_ids"]
    val_token_ent_b_masks: List[Tuple[int]] = val_tokenized_data_dict["token_entity_mask"]
    val_edge_index_token_idx: List[Tuple[int]] = val_tokenized_data_dict["edge_index_token_idx"]
    val_edge_index_entity_idx: List[Tuple[int]] = val_tokenized_data_dict["edge_index_entity_idx"]

    val_dataset = PrecomputedGraphTextDataset(tokenizer=sentence_bert_tokenizer,
                                              sentence_input_ids=val_sent_input_ids,
                                              token_ent_binary_masks=val_token_ent_b_masks,
                                              edge_index_token_idx=val_edge_index_token_idx,
                                              edge_index_entity_idx=val_edge_index_entity_idx,
                                              node_id2adjacency_list=node_id2adjacency_list,
                                              sentence_max_length=sentence_max_length,
                                              concept_max_length=concept_max_length,
                                              entity_masking_prob=entity_masking_probability,
                                              link_negative_sample_size=link_negative_sample_size,
                                              mlm_probability=mlm_probability,
                                              num_nodes=num_nodes,
                                              global2local_concept_id=global2local_concept_id,
                                              corrupt_sentences=args.corrupt_sentences,
                                              token_ent_mask_keep_one_ids_only=args.token_ent_mask_keep_one_ids_only)
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

    parser.add_argument("--graph_data_dir", type=str, )
    parser.add_argument("--bert_encoder_name")
    parser.add_argument("--train_tokenized_sentences_path", type=str)
    parser.add_argument("--val_tokenized_sentences_path", type=str, default=None, required=False)
    parser.add_argument("--tokenized_concepts_path", type=str)
    parser.add_argument("--sentence_max_length", type=int)
    parser.add_argument("--use_rel", action="store_true")
    parser.add_argument("--concept_max_length", type=int)
    # parser.add_argument("--masking_mode", choices=("text", "graph", "both", "random"), type=str)

    parser.add_argument("--mlm_probability", type=float)
    parser.add_argument("--entity_masking_probability", type=float)
    parser.add_argument("--link_negative_sample_size", type=int)
    parser.add_argument("--corrupt_sentences", action="store_true")
    parser.add_argument("--token_ent_mask_keep_one_ids_only", action="store_true")

    parser.add_argument('--dataloader_num_workers', type=int)
    parser.add_argument("--batch_size", type=int, )

    parser.add_argument("--train_output_dir", type=str, )
    parser.add_argument("--val_output_dir", type=str, )

    # -----------------------------------------------------

    # parser.add_argument("--graph_data_dir", type=str,
    #                     default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/graph_dataset_debug")
    # parser.add_argument("--bert_encoder_name", default="prajjwal1/bert-tiny")
    # parser.add_argument("--train_tokenized_sentences_path", type=str,
    #                     default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/graph_dataset_debug/v2_tokenized_sentences")
    # parser.add_argument("--val_tokenized_sentences_path", type=str, required=False,
    #                     default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/graph_dataset_debug/v2_tokenized_sentences")
    # parser.add_argument("--tokenized_concepts_path", type=str,
    #                     default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/graph_dataset_debug/node_id2terms_list_tokenized_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    # parser.add_argument("--sentence_max_length", type=int, default=128)
    # parser.add_argument("--use_rel", default=True)
    # parser.add_argument("--concept_max_length", type=int, default=32)
    # # parser.add_argument("--masking_mode", choices=("text", "graph", "both", "random"), type=str)
    #
    # parser.add_argument("--mlm_probability", type=float, default=0.15)
    # parser.add_argument("--entity_masking_probability", type=float, default=0.15)
    # parser.add_argument("--link_negative_sample_size", type=int, default=2)
    #
    # parser.add_argument("--corrupt_sentences", default=True)
    # parser.add_argument("--token_ent_mask_keep_one_ids_only", default=True)
    #
    # parser.add_argument('--dataloader_num_workers', type=int, default=2)
    # parser.add_argument("--batch_size", type=int, default=3)
    #
    # parser.add_argument("--train_output_dir", type=str, default="DELETE/train")
    # parser.add_argument("--val_output_dir", type=str, default="DELETE/val")

    args = parser.parse_args()
    main(args)
