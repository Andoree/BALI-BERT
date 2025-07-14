import logging
import os
from argparse import ArgumentParser
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
import gc

from textkb.data.precomputed_graph_embs_dataset import PrecomputedGraphTextDataset
from textkb.utils.io import load_tokenized_sentences_data_v2, load_adjacency_lists, \
    create_dir_if_not_exists, load_list_elem_per_line, load_tokenized_concepts, load_dict
from textkb.utils.umls2graph import filter_adjacency_lists
from textkb.utils.utils import token_ids2str, convert_input_ids_list_to_str_list


def create_batches(data_loader, tokenizer, debug_flag, output_dir: str, node_id2input_ids=None,
                   sanity_check_file_path=None, id2cui: Optional[Dict[int, str]] = None):
    mask_token_id = tokenizer.mask_token_id
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id
    dataset = data_loader.dataset
    global2local_concept_id = dataset.global2local_concept_id
    local2global_concept_id = {v: k for k, v in global2local_concept_id.items()}
    spec_token_ids = (mask_token_id, sep_token_id, cls_token_id, pad_token_id)
    NUM_DEBUG_SENTENCES = 3
    NUM_DEBUG_TOKEN_ENTITY_EDGES = 10
    NUM_DEBUG_POS_TRIPLES = 5
    if debug_flag:
        assert sanity_check_file_path is not None and node_id2input_ids is not None
        with open(sanity_check_file_path, 'w+', encoding="utf-8") as out_file:
            pass

    for i, batch in enumerate(tqdm(data_loader)):
        if i % 10000 == 0:
            print(gc.collect())
        sentence_input = [t for t in batch["sentence_input"]]
        sentence_input_ids, sentence_att_mask = sentence_input

        corr_sentence_input_ids, corr_sentence_att_mask, token_labels = None, None, None
        if batch.get("corrupted_sentence_input") is not None:
            corrupted_sentence_input = [t for t in batch["corrupted_sentence_input"]]
            corr_sentence_input_ids, corr_sentence_att_mask = corrupted_sentence_input
            token_labels = batch["token_labels"]

        entity_node_ids = batch["entity_node_ids"]
        token_node_ids = None
        if debug_flag:
            token_node_ids = batch["token_node_ids"]
        token_is_entity_mask = batch["token_is_entity_mask"]
        subtoken2entity_edge_index = batch["subtoken2entity_edge_index"]
        num_entities = batch["num_entities"]
        assert not sentence_input_ids.requires_grad

        pos_triples = batch["pos_triples"]
        neg_node_ids = batch["neg_node_ids"]
        has_edge_mask = batch["has_edge_mask"]
        batch_type = batch["batch_type"]
        token_ent_mask_keep_one_ids_only = batch["token_ent_mask_keep_one_ids_only"]
        assert neg_node_ids.dim() == 2
        if debug_flag:
            with open(sanity_check_file_path, 'a+', encoding="utf-8") as out_file:
                sentences = [token_ids2str(tokenizer, token_ids, spec_token_ids) for token_ids in
                             sentence_input_ids[:NUM_DEBUG_SENTENCES]]
                if token_ent_mask_keep_one_ids_only:
                    tok_e_mask = torch.zeros(size=sentence_input_ids.size(), dtype=torch.long)
                    for j, one_ids in enumerate(token_is_entity_mask):
                        tok_e_mask[j, one_ids] = 1
                else:
                    tok_e_mask = token_is_entity_mask

                e_token_ids = sentence_input_ids[tok_e_mask > 0]
                assert len(e_token_ids) == tok_e_mask.sum()
                assert len(token_node_ids) == len(subtoken2entity_edge_index[0])
                ei_subtoken_index = subtoken2entity_edge_index[0][:NUM_DEBUG_TOKEN_ENTITY_EDGES]
                # ei_entity_index = subtoken2entity_edge_index[1][:100].tolist()
                ei_entity_index = token_node_ids[:NUM_DEBUG_TOKEN_ENTITY_EDGES]

                edge_subtoken_input_ids = e_token_ids[ei_subtoken_index]
                ei_entity_index_input_ids = [node_id2input_ids[local2global_concept_id[x]][0]
                                             for x in ei_entity_index]

                edge_subtoken_str = tokenizer.convert_ids_to_tokens(edge_subtoken_input_ids)
                edge_entity_str = convert_input_ids_list_to_str_list(ei_entity_index_input_ids, tokenizer,
                                                                     spec_token_ids)

                sentences_str = '\n\t'.join(sentences)
                sentences_str = f"Sentences {batch_type}:\n\t{sentences_str}\n"
                token_entity_edges_str = '\n\t'.join(f"{e} -- {n}" for e, n
                                                     in zip(edge_subtoken_str, edge_entity_str))
                token_entity_edges_str = f"Edges {batch_type}:\n\t{token_entity_edges_str}\n"

                pos_triples_heads = pos_triples[0]
                pos_triples_tails = pos_triples[2]
                assert len(pos_triples_heads) == len(pos_triples_tails) == len(entity_node_ids[has_edge_mask > 0])
                pos_triples_heads = pos_triples_heads[:NUM_DEBUG_POS_TRIPLES]
                pos_triples_tails = pos_triples_tails[:NUM_DEBUG_POS_TRIPLES]
                triple_head_input_ids = [node_id2input_ids[local2global_concept_id[x.item()]][0] for x in
                                         pos_triples_heads]  # [:20]]
                triple_tail_input_ids = [node_id2input_ids[local2global_concept_id[x.item()]][0] for x in
                                         pos_triples_tails]  # [:20]]
                triple_head_cuis = [id2cui[local2global_concept_id[x.item()]] for x in pos_triples_heads]  # [:20]]
                triple_tail_cuis = [id2cui[local2global_concept_id[x.item()]] for x in pos_triples_tails]  # [:20]]

                assert len(entity_node_ids) == len(has_edge_mask)

                triple_head_tokens = \
                    [tokenizer.convert_ids_to_tokens([y for y in x if y not in spec_token_ids]) for x in
                     triple_head_input_ids]
                triple_head_tokens = ["".join((x.strip("#") if x.startswith("#") else f" {x}" for x in t)) for t in
                                      triple_head_tokens]

                triple_tail_tokens = \
                    [tokenizer.convert_ids_to_tokens([y for y in x if y not in spec_token_ids]) for x in
                     triple_tail_input_ids]
                triple_tail_tokens = ["".join((x.strip("#") if x.startswith("#") else f" {x}" for x in t)) for t in
                                      triple_tail_tokens]
                triples_str = '\n\t'.join(f"{h} ({hc}) --> {t} ({tc})" for h, t, hc, tc in
                                          zip(triple_head_tokens, triple_tail_tokens, triple_head_cuis,
                                              triple_tail_cuis))
                triples_str = f"Triples {batch_type}:\n\t{triples_str}\n"
                out_file.write(f"{sentences_str}{token_entity_edges_str}{triples_str}", )
                out_file.flush()

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
    graph_data_dir = args.graph_data_dir
    adjacency_lists_path = os.path.join(graph_data_dir, "adjacency_lists")
    id2cui_path = os.path.join(graph_data_dir, "id2cui")
    id2cui = load_dict(id2cui_path, dtype_1=int, dtype_2=str)

    use_rel = args.use_rel
    drop_selfloops = args.drop_selfloops
    sentence_max_length = args.sentence_max_length
    # masking_mode = args.masking_mode
    debug = args.debug
    sanity_check_file_path = None
    node_id2input_ids: Optional[List[Tuple[Tuple[int]]]] = None
    if debug:
        tokenized_concepts_path = args.tokenized_concepts_path
        sanity_check_file_path = args.sanity_check_file_path
        node_id2input_ids = load_tokenized_concepts(tok_conc_path=tokenized_concepts_path)

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

    node_id2adjacency_list = load_adjacency_lists(adjacency_lists_path, use_rel=use_rel, drop_selfloops=drop_selfloops)

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
                                                entity_masking_prob=entity_masking_probability,
                                                link_negative_sample_size=link_negative_sample_size,
                                                mlm_probability=mlm_probability,
                                                global2local_concept_id=global2local_concept_id,
                                                num_nodes=num_nodes,
                                                corrupt_sentences=args.corrupt_sentences,
                                                token_ent_mask_keep_one_ids_only=args.token_ent_mask_keep_one_ids_only,
                                                debug=debug)

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
                                              entity_masking_prob=entity_masking_probability,
                                              link_negative_sample_size=link_negative_sample_size,
                                              mlm_probability=mlm_probability,
                                              num_nodes=num_nodes,
                                              global2local_concept_id=global2local_concept_id,
                                              corrupt_sentences=args.corrupt_sentences,
                                              token_ent_mask_keep_one_ids_only=args.token_ent_mask_keep_one_ids_only,
                                              debug=debug)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.dataloader_num_workers,
                                shuffle=False, collate_fn=val_dataset.collate_fn)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.dataloader_num_workers,
                                  shuffle=True, collate_fn=train_dataset.collate_fn)

    create_batches(data_loader=train_dataloader,
                   tokenizer=sentence_bert_tokenizer,
                   debug_flag=debug,
                   node_id2input_ids=node_id2input_ids,
                   id2cui=id2cui,
                   sanity_check_file_path=sanity_check_file_path,
                   output_dir=train_output_dir)
    create_batches(data_loader=val_dataloader,
                   tokenizer=sentence_bert_tokenizer,
                   debug_flag=debug,
                   node_id2input_ids=node_id2input_ids,
                   id2cui=id2cui,
                   sanity_check_file_path=sanity_check_file_path,
                   output_dir=val_output_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()

    parser.add_argument("--graph_data_dir", type=str, )
    parser.add_argument("--bert_encoder_name")
    parser.add_argument("--train_tokenized_sentences_path", type=str)
    parser.add_argument("--val_tokenized_sentences_path", type=str, default=None, required=False)
    parser.add_argument("--tokenized_concepts_path", required=False)
    parser.add_argument("--sentence_max_length", type=int)
    parser.add_argument("--use_rel", action="store_true")
    parser.add_argument("--drop_selfloops", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sanity_check_file_path", required=False)

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
    # parser.add_argument("--bert_encoder_name",
    #                     default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    # # default="prajjwal1/bert-tiny")
    # parser.add_argument("--train_tokenized_sentences_path", type=str,
    #                     default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/graph_dataset_debug/v2_tokenized_sentences")
    # parser.add_argument("--val_tokenized_sentences_path", type=str, required=False,
    #                     default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/graph_dataset_debug/v2_tokenized_sentences")
    # parser.add_argument("--tokenized_concepts_path", type=str,
    #                     default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/graph_dataset_debug/node_id2terms_list_tokenized_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    # parser.add_argument("--sentence_max_length", type=int, default=128)
    # parser.add_argument("--use_rel", default=True)
    # parser.add_argument("--drop_selfloops", default=True)
    # parser.add_argument("--debug", default=True)
    # parser.add_argument("--sanity_check_file_path",
    #                     default="DELETE/dataset_sanity_check.txt")
    # # parser.add_argument("--masking_mode", choices=("text", "graph", "both", "random"), type=str)
    #
    # parser.add_argument("--mlm_probability", type=float, default=0.15)
    # parser.add_argument("--entity_masking_probability", type=float, default=0.0)
    # parser.add_argument("--link_negative_sample_size", type=int, default=2)
    #
    # parser.add_argument("--corrupt_sentences", default=True)
    # parser.add_argument("--token_ent_mask_keep_one_ids_only", default=True)
    #
    # parser.add_argument('--dataloader_num_workers', type=int, default=0)
    # parser.add_argument("--batch_size", type=int, default=8)
    #
    # parser.add_argument("--train_output_dir", type=str, default="DELETE/train")
    # parser.add_argument("--val_output_dir", type=str, default="DELETE/val")

    args = parser.parse_args()
    main(args)
