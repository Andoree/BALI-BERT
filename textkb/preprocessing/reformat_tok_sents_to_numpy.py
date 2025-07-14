import logging
import os
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from textkb.utils.io import create_dir_if_not_exists


def reformat(input_tokenized_data_dir: str, output_tokenized_dir: str, field_sep: str = '\t'):
    num_files = len(list(os.listdir(input_tokenized_data_dir)))
    for fname in tqdm(os.listdir(input_tokenized_data_dir), total=num_files, mininterval=10.0):
        if fname == "config.txt":
            continue
        batch_size = 100
        i = 0
        input_fpath = os.path.join(input_tokenized_data_dir, fname)

        fname_no_ext = fname.split('.')[0]
        with open(input_fpath, 'r', encoding="utf-8") as inp_file:
            file_data = []
            for line in inp_file:
                attrs = line.strip().split(field_sep)

                inp_ids = tuple(int(x) for x in attrs[1].split(','))
                token_mask = tuple(int(x) for x in attrs[2].split(','))
                edge_index_token_idx = tuple(int(x) for x in attrs[3].split(','))
                edge_index_entity_idx = tuple(int(x) for x in attrs[4].split(','))

                inp_ids_end = len(inp_ids)
                token_mask_end = inp_ids_end + len(token_mask)
                edge_index_token_idx_end = token_mask_end + len(edge_index_token_idx)
                edge_index_entity_idx_end = edge_index_token_idx_end + len(edge_index_entity_idx)

                spans = (inp_ids_end, token_mask_end, edge_index_token_idx_end, edge_index_entity_idx_end)
                lst = spans + inp_ids + token_mask + edge_index_token_idx + edge_index_entity_idx
                # lst_s = ','.join((str(x) for x in lst))
                # spans_s = f"{inp_ids_end},{token_mask_end},{edge_index_token_idx_end},{edge_index_entity_idx_end}"
                # out_file.write(f"{spans_s},{lst_s}\n")

                # new_line = field_sep.join(attrs[1:])
                # out_file.write(f"{new_line}\n")
                file_data.append(lst)
                if len(file_data) > batch_size:
                    max_length = max(len(t) for t in file_data)
                    num_samples = len(file_data)
                    np_array = np.zeros(shape=(num_samples, max_length), dtype=np.int32)
                    output_fpath = os.path.join(output_tokenized_dir, f"{fname_no_ext}_{i}.npy")
                    np.save(output_fpath, np_array)
                    file_data.clear()
                    i += 1

            if len(file_data) > batch_size:
                max_length = max(len(t) for t in file_data)
                num_samples = len(file_data)
                np_array = np.zeros(shape=(num_samples, max_length), dtype=np.int32)
                output_fpath = os.path.join(output_tokenized_dir, f"{fname_no_ext}_{i}.npy")
                np.save(output_fpath, np_array)
                file_data.clear()


def main(args):
    tokenized_sentences_dir = args.tokenized_sentences_dir
    output_dir = args.output_dir
    create_dir_if_not_exists(output_dir)

    reformat(input_tokenized_data_dir=tokenized_sentences_dir,
             output_tokenized_dir=output_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument("--tokenized_sentences_dir", type=str,
                        default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/graph_dataset_debug/tokenized_sentences")
    parser.add_argument("--output_dir", type=str,
                        default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/graph_dataset_debug/numpy_tokenized_sentences")

    args = parser.parse_args()
    main(args)
