import logging
import os
import random
import re
from argparse import ArgumentParser
from typing import Dict, List, Set

from tqdm import tqdm

from textkb.preprocessing.create_data_index import DATA_FILENAME_PATTERN
from textkb.utils.io import create_dir_if_not_exists


def create_entity_balanced_dataset(input_tokenized_data_dir: str,
                                   max_concept_frequency: int,
                                   output_tokenized_data_dir: str):
    num_files = len(os.listdir(input_tokenized_data_dir))
    entity_id2line_ids: Dict[int, List[int]] = {}
    line_id = 0
    logging.info(f"Reading data, creating entity2sample_id index from:\n{input_tokenized_data_dir}")
    for fname in tqdm(sorted(os.listdir(input_tokenized_data_dir)), total=num_files):
        m = re.fullmatch(DATA_FILENAME_PATTERN, fname)
        if m:
            input_fpath = os.path.join(input_tokenized_data_dir, fname)
            with open(input_fpath, 'r', encoding="utf-8") as inp_file:
                for line in inp_file:
                    # Read sample
                    data = tuple(map(int, line.strip().split(',')))
                    inp_ids_end, token_mask_end, ei_tok_idx_end, ei_ent_idx_end = data[:4]
                    # sentence_input_ids = data[4:inp_ids_end + 4]
                    # token_entity_mask = data[inp_ids_end + 4:token_mask_end + 4]
                    # edge_index_token_idx = data[token_mask_end + 4:ei_tok_idx_end + 4]

                    edge_index_entity_idx = data[ei_tok_idx_end + 4:ei_ent_idx_end + 4]
                    # Find mentioned entities, make a set from them
                    edge_index_entity_idx = set(edge_index_entity_idx)
                    # Update index: entity_id2line_id: sample where entity is mentioned
                    for e_id in edge_index_entity_idx:
                        if entity_id2line_ids.get(e_id) is None:
                            entity_id2line_ids[e_id] = []
                        entity_id2line_ids[e_id].append(line_id)

                    line_id += 1

    global_selected_line_ids: Set[int] = set()
    logging.info("Subsampling data...")
    # Fill global_selected_line_ids: truncated sampling of samples entity-wise
    for entity_id, line_ids in tqdm(entity_id2line_ids.items(), total=len(entity_id2line_ids)):
        num_samples_to_keep = min(len(line_ids), max_concept_frequency)
        sampled_line_ids = random.sample(line_ids, num_samples_to_keep)
        global_selected_line_ids.update(sampled_line_ids)

    logging.info(f"Finished sampling. Kept {len(global_selected_line_ids)} samples in total")

    line_id = 0
    logging.info("Writing output dataset...")
    for fname in tqdm(sorted(os.listdir(input_tokenized_data_dir)), total=num_files):

        m = re.fullmatch(DATA_FILENAME_PATTERN, fname)
        if m:
            input_fpath = os.path.join(input_tokenized_data_dir, fname)
            output_fpath = os.path.join(output_tokenized_data_dir, fname)

            with open(input_fpath, 'r', encoding="utf-8") as inp_file, \
                    open(output_fpath, 'w+', encoding="utf-8") as out_file:
                for line in inp_file:
                    if line_id in global_selected_line_ids:
                        out_file.write(line)
                    line_id += 1


def main(args):
    input_tokenized_data_dir = args.input_tokenized_data_dir
    max_concept_frequency = args.max_concept_frequency
    output_data_dir = args.output_data_dir
    create_dir_if_not_exists(output_data_dir)

    create_entity_balanced_dataset(input_tokenized_data_dir=input_tokenized_data_dir,
                                   max_concept_frequency=max_concept_frequency,
                                   output_tokenized_data_dir=output_data_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--input_tokenized_data_dir', type=str)
    parser.add_argument('--max_concept_frequency', type=int, )
    parser.add_argument('--output_data_dir', type=str, )
    args = parser.parse_args()
    main(args)
