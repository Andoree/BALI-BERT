import logging
import os
import re
from argparse import ArgumentParser
from collections import Counter

from tqdm import tqdm
from transformers import AutoTokenizer

from textkb.utils.io import create_dir_if_not_exists, load_offset_index, untokenize_concept_names


def main(args):
    input_data_dir = args.input_data_dir
    tokenized_concepts_path = args.tokenized_concepts_path
    tokenizer_name_or_path = args.tokenizer_name_or_path
    output_data_dir = args.output_data_dir

    create_dir_if_not_exists(output_data_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    logging.info("Loading tokenized concept names...")
    concept_names = untokenize_concept_names(path=tokenized_concepts_path, tokenizer=tokenizer)
    logging.info(f"Loaded {len(concept_names)} tokenized concept names...")

    logging.info(f"Loading offset index...")
    offsets, offset_lowerbounds, offset_upperbounds, offset_filenames = load_offset_index(input_data_dir,
                                                                                          prefix="train")
    logging.info(f"Loaded offset index: {len(offsets)} samples...")

    filename = None
    counter = Counter()
    logging.info(f"Processing data....")
    for idx, offset in tqdm(enumerate(offsets)):
        for j, (lb, ub) in enumerate(zip(offset_lowerbounds, offset_upperbounds)):
            if lb <= idx < ub:
                filename = offset_filenames[j]
                break
        assert filename is not None
        fpath = os.path.join(input_data_dir, filename)

        with open(fpath, 'r', encoding="utf-8") as inp_file:
            inp_file.seek(offset)
            line = inp_file.readline()

            data = tuple(map(int, line.strip().split(',')))
            inp_ids_end, token_mask_end, ei_tok_idx_end, ei_ent_idx_end = data[:4]
            sentence_input_ids = data[4:inp_ids_end + 4]
            token_entity_mask = data[inp_ids_end + 4:token_mask_end + 4]
            # Token indices in mentions
            edge_index_token_idx = data[token_mask_end + 4:ei_tok_idx_end + 4]
            # Ids of entities mentioned in text
            edge_index_entity_idx = data[ei_tok_idx_end + 4:ei_ent_idx_end + 4]
            edge_index_entity_idx = set(edge_index_entity_idx)

            for e_id in edge_index_entity_idx:
                counter[e_id] += 1
    for e_id, count in counter.most_common(10):
        name = concept_names[e_id]
        logging.info(f"{e_id} ({name}) : {count}\n")

    output_stats_path = os.path.join(output_data_dir, "stats_entity_distribution.txt")
    with open(output_stats_path, 'w+', encoding="utf-8") as out_file:
        for e_id, count in counter.most_common(len(tuple(counter.keys()))):
            name = concept_names[e_id]
            out_file.write(f"{e_id} ({name}) : {count}\n")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--input_data_dir')
    parser.add_argument('--tokenized_concepts_path')
    parser.add_argument('--tokenizer_name_or_path')
    parser.add_argument('--output_data_dir')

    args = parser.parse_args()
    main(args)
