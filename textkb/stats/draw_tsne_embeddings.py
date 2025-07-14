import logging
import os
from argparse import ArgumentParser
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from textkb.data.offsets_dataset import TextGraphGraphNeighborsOffsetDataset
from textkb.utils.io import create_dir_if_not_exists, load_offset_index, load_tokenized_concepts, load_adjacency_lists
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def infer_embeddings(bert_encoder, dataloader, batch_count, device):
    embeddings_list = []
    embedding_labels = []
    hidden_size = bert_encoder.config.hidden_size
    bert_encoder.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=batch_count):
            if i > batch_count:
                break
            sentence_input = [t.to(device) for t in batch["corrupted_sentence_input"]]
            token_is_entity_mask = batch["token_is_entity_mask"].to(device)
            input_ids, att_mask = sentence_input

            text_emb = bert_encoder(input_ids, attention_mask=att_mask,
                                    return_dict=True)['last_hidden_state']
            assert text_emb.dim() == 3
            assert text_emb.size(2) == hidden_size

            token_embs = text_emb[att_mask > 0, :]
            entity_mask = token_is_entity_mask[att_mask > 0]
            assert len(token_embs) == len(entity_mask)

            embeddings_list.extend(token_embs.cpu().detach().tolist())
            embedding_labels.extend(entity_mask.cpu().detach().tolist())
    tsne = TSNE(n_components=2, random_state=29, n_iter=15000, metric="cosine")
    embeddings_list = np.array(embeddings_list)
    embedding_labels = np.array(embedding_labels)

    embs_2d = tsne.fit_transform(embeddings_list)

    return embs_2d, embedding_labels


def draw_labeled_embeddings(embs_2d, emb_labels, output_path):
    COLOR_DICT = {0: "red", 1: "green"}
    figsize = (10, 8)
    plt.figure(figsize=figsize)

    x = embs_2d[:, 0]
    y = embs_2d[:, 1]
    print(emb_labels)
    logging.info(f"emb_labels {emb_labels}")
    emb_labels = list(map(lambda z: COLOR_DICT[z], emb_labels))
    plt.scatter(x, y, c=emb_labels, )
    plt.savefig(output_path, format="png")


def main(args):
    model_name_or_path = args.model_name_or_path
    data_dir = args.data_dir
    graph_data_dir = args.graph_data_dir
    sample_size = args.sample_size
    tokenized_concepts_path = args.tokenized_concepts_path
    output_path = args.output_path

    output_dir = os.path.dirname(output_path)
    create_dir_if_not_exists(output_dir)

    MAX_N_NEIGHBORS = 3
    SENTENCE_MAX_LENGTH = 128
    CONCEPT_MAX_LENGTH = 32
    MLM_PROBABILITY = 0.
    GRAPH_FORMAT = "edge_index"
    GRAPH_MLM_TASK = False
    LIN_GRAPH_MAX_LENGTH = None
    MENTION_MASK_PROB = 0.
    CONCEPT_MASK_PROB = 0.
    TOKEN_ENTITY_INDEX_TYPE = "edge_index"
    LINEAR_GRAPH_FORMAT = "v1"
    BATCH_SIZE = 4

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    bert_encoder = AutoModel.from_pretrained(model_name_or_path)
    device = "cpu"
    bert_encoder.to(device)

    create_dir_if_not_exists(output_dir)
    offsets, offset_lowerbounds, offset_upperbounds, offset_filenames = load_offset_index(data_dir, prefix="val")
    adjacency_lists_path = os.path.join(graph_data_dir, "adjacency_lists")

    node_id2input_ids: List[Tuple[Tuple[int]]] = load_tokenized_concepts(tok_conc_path=tokenized_concepts_path)

    node_id2adjacency_list = load_adjacency_lists(adjacency_lists_path,
                                                  use_rel=False,
                                                  drop_selfloops=False,
                                                  use_rel_or_rela="rel")

    dataset = TextGraphGraphNeighborsOffsetDataset(tokenizer=tokenizer,
                                                   input_data_dir=data_dir,
                                                   offsets=offsets,
                                                   offset_lowerbounds=offset_lowerbounds,
                                                   offset_upperbounds=offset_upperbounds,
                                                   offset_filenames=offset_filenames,
                                                   node_id2adjacency_list=node_id2adjacency_list,
                                                   node_id2input_ids=node_id2input_ids,
                                                   max_n_neighbors=MAX_N_NEIGHBORS,
                                                   use_rel=False,
                                                   rel_id2tokenized_name=None,
                                                   sentence_max_length=SENTENCE_MAX_LENGTH,
                                                   concept_max_length=CONCEPT_MAX_LENGTH,
                                                   mlm_probability=MLM_PROBABILITY,
                                                   graph_format=GRAPH_FORMAT,
                                                   linear_graph_format=LINEAR_GRAPH_FORMAT,
                                                   graph_mlm_task=GRAPH_MLM_TASK,
                                                   lin_graph_max_length=LIN_GRAPH_MAX_LENGTH,
                                                   mention_masking_prob=MENTION_MASK_PROB,
                                                   concept_name_masking_prob=CONCEPT_MASK_PROB,
                                                   token_entity_index_type=TOKEN_ENTITY_INDEX_TYPE)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                             num_workers=2,
                             shuffle=False, collate_fn=dataset.collate_fn)
    analyze_batch_count = sample_size // BATCH_SIZE
    embs_2d, entity_mask = infer_embeddings(bert_encoder, data_loader, analyze_batch_count, device)
    logging.info(f"entity_mask {type(entity_mask)}")
    logging.info(f"entity_mask {entity_mask}")
    print("entity_mask", entity_mask)
    draw_labeled_embeddings(embs_2d=embs_2d, emb_labels=entity_mask, output_path=output_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )

    parser = ArgumentParser()
    # parser.add_argument('--model_name_or_path', type=str, required=True)
    # parser.add_argument('--data_dir', type=str, required=True)
    # parser.add_argument('--graph_data_dir', type=str, required=True)
    # parser.add_argument('--sample_size', type=int, required=True)
    # parser.add_argument('--tokenized_concepts_path', type=str, required=True)
    # parser.add_argument('--output_path', type=str, required=True)

    # parser.add_argument('--model_name_or_path', type=str, required=False,
    #                     default="prajjwal1/bert-tiny")
    # parser.add_argument('--data_dir', type=str, required=False,
    #                     default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/2024_feb_debug_dataset/tokenized_sentences")
    # parser.add_argument('--graph_data_dir', type=str, required=False,
    #                     default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/2024_feb_debug_dataset/graph/")
    # parser.add_argument('--sample_size', type=int, required=False,
    #                     default=100)
    # parser.add_argument('--tokenized_concepts_path', type=str, required=False,
    #                     default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/2024_feb_debug_dataset/graph/node_id2terms_list_tokenized_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    # parser.add_argument('--output_path', type=str, required=False,
    #                     default="./tsne_figures/test.png")

    # parser.add_argument('--model_name_or_path', type=str, required=False,
    #                     default="/home/c204/University/NLP/text_kb/alignment_model_for_tsne/alignment_model_tgcl/")
    parser.add_argument('--model_name_or_path', type=str, required=False,
                        default="/home/c204/University/NLP/text_kb/alignment_model_for_tsne/alignment_model/")
                        # default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument('--data_dir', type=str, required=False,
                        default="/home/c204/University/NLP/text_kb/alignment_model_for_tsne/alignment_data/")
    parser.add_argument('--graph_data_dir', type=str, required=False,
                        default="/home/c204/University/NLP/text_kb/alignment_model_for_tsne/graph_dataset/")
    parser.add_argument('--sample_size', type=int, required=False,
                        default=100)
    parser.add_argument('--tokenized_concepts_path', type=str, required=False,
                        default="/home/c204/University/NLP/text_kb/alignment_model_for_tsne/graph_dataset/node_id2terms_list_tokenized_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument('--output_path', type=str, required=False,
                        default="./tsne_figures/alignment_model_tgcl.png")

    args = parser.parse_args()
    main(args)
