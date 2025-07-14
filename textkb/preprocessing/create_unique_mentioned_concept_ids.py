import logging
import os
from argparse import ArgumentParser

from textkb.utils.io import create_dir_if_not_exists, get_unique_mentioned_concept_ids_from_data, \
    save_list_elem_per_line, load_adjacency_lists


def main(args):
    input_tokenized_sentences_dirs = args.input_tokenized_sentences_dirs
    input_graph_data_dir = args.graph_data_dir
    output_graph_dataset_dir = args.output_graph_dataset_dir
    create_dir_if_not_exists(output_graph_dataset_dir)
    concept_ids_set = set()
    for tokenized_data_dir in input_tokenized_sentences_dirs:
        logging.info(f"Processing {tokenized_data_dir}")
        concept_ids_set.update(get_unique_mentioned_concept_ids_from_data(tokenized_data_dir))
    logging.info(f"Finished processing all data. {len(concept_ids_set)} unique concept ids mentioned")

    adjacency_lists_path = os.path.join(input_graph_data_dir, "adjacency_lists")
    node_id2adjacency_list = load_adjacency_lists(adjacency_lists_path, False)

    node_ids_with_an_edge = set(node_id2adjacency_list.keys())
    logging.info(f"Num nodes with an edge: {len(node_ids_with_an_edge)}")
    inter = concept_ids_set.intersection(node_ids_with_an_edge)
    union = concept_ids_set.union(node_ids_with_an_edge)
    logging.info(f"Intersection of nodes with an edge and mentioned nodes: {len(inter)}")
    logging.info(f"Union of nodes with an edge and mentioned nodes: {len(union)}")
    concept_ids_set = concept_ids_set.union(node_ids_with_an_edge)

    unique_concept_idx_path = os.path.join(output_graph_dataset_dir, "mentioned_concepts_idx")
    save_list_elem_per_line(lst=concept_ids_set, output_path=unique_concept_idx_path)
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--input_tokenized_sentences_dirs',
                        nargs='+', default=("/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/graph_dataset_debug/v2_tokenized_sentences",
                                            "/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/graph_dataset_debug/v2_tokenized_sentences"))
    parser.add_argument('--graph_data_dir', default="/home/c204/University/NLP/BERN2_sample/debug_graph")
    parser.add_argument('--output_graph_dataset_dir', type=str,
                        default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/graph_dataset_debug/")
    args = parser.parse_args()
    main(args)
