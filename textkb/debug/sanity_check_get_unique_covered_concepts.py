import logging
import os
import random
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from textkb.data.offsets_dataset import TextGraphGraphNeighborsOffsetDataset
from textkb.utils.io import load_list_elem_per_line, \
    load_tokenized_concepts, load_offset_index, load_adjacency_lists, load_dict, create_dir_if_not_exists


def print_unique_entity_node_ids(data_loader, out_path):
    pbar = tqdm(data_loader, miniters=len(data_loader) // 100, total=len(data_loader))
    unique_entity_node_ids = set()
    with open(out_path, 'w', encoding="utf-8") as out_file:
        for in_batch_step_id, batch in enumerate(pbar):
            entity_node_ids = batch["entity_node_ids"]
            entity_node_ids = entity_node_ids.tolist()

            unique_entity_node_ids.update(entity_node_ids)
        print(f"{len(unique_entity_node_ids)}")
        out_file.write(f"{len(unique_entity_node_ids)}\n")
        for en_id in unique_entity_node_ids:
            out_file.write(f"{en_id}\n")


def main(args):
    transformer_tokenizer = AutoTokenizer.from_pretrained(args.transformer_tokenizer_name)

    node_id2terms_path = args.node_id2terms_path
    graph_data_dir = args.graph_data_dir
    node_id2tokenized_terms = load_tokenized_concepts(node_id2terms_path)
    token_entity_index_type = args.token_entity_index_type
    tokenized_data_dir = args.tokenized_data_dir
    graph_format = args.graph_format
    use_rel_or_rela = args.use_rel_or_rela
    rela2rela_name_path = args.rela2rela_name
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    create_dir_if_not_exists(output_dir)
    if use_rel_or_rela == "rel":
        rel2id_path = os.path.join(graph_data_dir, "rel2id")
        rel2id = load_dict(rel2id_path, dtype_1=str, dtype_2=int)
    elif use_rel_or_rela == "rela":
        rela2id_path = os.path.join(graph_data_dir, "rela2id")
        rel2id = load_dict(rela2id_path, dtype_1=str, dtype_2=int)
    else:
        raise ValueError(f"Invalid use_rel_or_rela : {use_rel_or_rela}")
    id2rel = {v: k for k, v in rel2id.items()}

    MAX_N_NEIGHBORS = 3
    USE_REL = True
    SENTENCE_MAX_LENGTH = 128
    CONCEPT_MAX_LENGTH = 32
    MLM_PROBABILITY = 0.0
    MENTION_MASKING_PROB = 0.0
    CONCEPT_NAME_MASKING_PROB = 0.0
    lin_graph_max_length = 256
    BATCH_SIZE = args.batch_size

    output_fname = os.path.basename(output_path)
    output_dir = os.path.dirname(output_path)
    output_train_fname = f"train_{output_fname}"
    output_val_fname = f"val_{output_fname}"
    output_train_path = os.path.join(output_dir, output_train_fname)
    output_val_path = os.path.join(output_dir, output_val_fname)

    rela2id_path = os.path.join(graph_data_dir, "rela2id")
    rela2id = load_dict(rela2id_path, dtype_1=str, dtype_2=int)

    rel_id2tokenized_name = None
    if rela2rela_name_path is not None:
        rel2rel_name = load_dict(rela2rela_name_path, dtype_1=str, dtype_2=str)
        print("rela2id", rela2id)
        # rel_id2name = {rela2id[k]: v for k, v in rel2rel_name.items()}
        rel_id2name = {v: rel2rel_name[k] for k, v in rela2id.items()}
        rel_id2tokenized_name = {k: tuple(transformer_tokenizer.encode_plus(v,
                                                                            max_length=16,
                                                                            add_special_tokens=False,
                                                                            truncation=True)["input_ids"])
                                 for k, v in rel_id2name.items()}

    adjacency_lists_path = os.path.join(graph_data_dir, "adjacency_lists")
    node_id2adjacency_list = load_adjacency_lists(adjacency_lists_path, USE_REL, drop_selfloops=True,
                                                  use_rel_or_rela=use_rel_or_rela)
    logging.info(f"Processing training set...")
    tr_offsets, tr_offset_lowerbounds, tr_offset_upperbounds, tr_offset_filenames = load_offset_index(
        tokenized_data_dir,
        prefix="train")
    # TODO

    tr_dataset = TextGraphGraphNeighborsOffsetDataset(tokenizer=transformer_tokenizer,
                                                      input_data_dir=tokenized_data_dir,
                                                      offsets=tr_offsets,
                                                      offset_lowerbounds=tr_offset_lowerbounds,
                                                      offset_upperbounds=tr_offset_upperbounds,
                                                      offset_filenames=tr_offset_filenames,
                                                      node_id2adjacency_list=node_id2adjacency_list,
                                                      node_id2input_ids=node_id2tokenized_terms,
                                                      max_n_neighbors=MAX_N_NEIGHBORS,
                                                      token_entity_index_type=token_entity_index_type,
                                                      use_rel=USE_REL,
                                                      graph_format=graph_format,
                                                      rel_id2tokenized_name=rel_id2tokenized_name,
                                                      lin_graph_max_length=lin_graph_max_length,
                                                      sentence_max_length=SENTENCE_MAX_LENGTH,
                                                      concept_max_length=CONCEPT_MAX_LENGTH,
                                                      mlm_probability=MLM_PROBABILITY,
                                                      concept_name_masking_prob=CONCEPT_NAME_MASKING_PROB,
                                                      mention_masking_prob=MENTION_MASKING_PROB)
    tr_data_loader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, num_workers=0,
                                shuffle=True, collate_fn=tr_dataset.collate_fn)
    print_unique_entity_node_ids(data_loader=tr_data_loader,
                                 out_path=output_train_path)

    ###############################################################################
    ###############################################################################
    logging.info(f"Processing validation set...")
    val_offsets, val_offset_lowerbounds, val_offset_upperbounds, val_offset_filenames = load_offset_index(
        tokenized_data_dir,
        prefix="val")
    val_dataset = TextGraphGraphNeighborsOffsetDataset(tokenizer=transformer_tokenizer,
                                                       input_data_dir=tokenized_data_dir,
                                                       offsets=val_offsets,
                                                       offset_lowerbounds=val_offset_lowerbounds,
                                                       offset_upperbounds=val_offset_upperbounds,
                                                       offset_filenames=val_offset_filenames,
                                                       node_id2adjacency_list=node_id2adjacency_list,
                                                       node_id2input_ids=node_id2tokenized_terms,
                                                       max_n_neighbors=MAX_N_NEIGHBORS,
                                                       token_entity_index_type=token_entity_index_type,
                                                       use_rel=USE_REL,
                                                       graph_format=graph_format,
                                                       lin_graph_max_length=lin_graph_max_length,
                                                       rel_id2tokenized_name=rel_id2tokenized_name,
                                                       sentence_max_length=SENTENCE_MAX_LENGTH,
                                                       concept_max_length=CONCEPT_MAX_LENGTH,
                                                       mlm_probability=MLM_PROBABILITY,
                                                       concept_name_masking_prob=CONCEPT_NAME_MASKING_PROB,
                                                       mention_masking_prob=MENTION_MASKING_PROB)

    val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0,
                                 shuffle=True, collate_fn=val_dataset.collate_fn)

    print_unique_entity_node_ids(data_loader=val_data_loader,
                                 out_path=output_val_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )

    parser = ArgumentParser()
    # parser.add_argument('--tokenized_data_dir',
    #                     default="/home/c204/University/NLP/BERN2_sample/TOKENIZED_UNMASKED")
    # parser.add_argument('--node_id2cui_path',
    #                     default="/home/c204/University/NLP/BERN2_sample/debug_graph/id2cui")
    # parser.add_argument('--transformer_tokenizer_name', default="prajjwal1/bert-tiny")
    # parser.add_argument('--node_id2terms_path',
    #                     default="/home/c204/University/NLP/BERN2_sample/debug_graph/node_id2terms_list")

    parser.add_argument('--tokenized_data_dir',
                        default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/2024_feb_debug_dataset/tokenized_sentences/")
    parser.add_argument('--graph_data_dir',
                        default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/2024_feb_debug_dataset/graph/")
    parser.add_argument('--transformer_tokenizer_name',
                        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument('--batch_size', default=32, type=int, required=False)
    parser.add_argument("--token_entity_index_type", type=str, choices=("edge_index", "matrix"),
                        required=False, default="edge_index")
    parser.add_argument("--graph_format", type=str, choices=("edge_index", "linear"),
                        required=False, default="edge_index")
    parser.add_argument("--use_rel_or_rela", type=str, required=False, choices=("rel", "rela"),
                        default="rel")
    parser.add_argument("--rela2rela_name", type=str, required=False)
    # parser.add_argument("--rela2rela_name", type=str, required=False,
    #                     default="/home/c204/University/NLP/text_kb/rela2rela_name_2020ab.tsv")
    # default="prajjwal1/bert-tiny")
    parser.add_argument('--node_id2terms_path',
                        default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/2024_feb_debug_dataset/graph/node_id2terms_list_tokenized_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument('--output_path',
                        default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/2024_feb_debug_dataset/sanity_check_offset_dataset_edge_index_graph_token2entity.txt")

    args = parser.parse_args()
    main(args)
