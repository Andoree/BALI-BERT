import logging
import os
import random
import re
from argparse import ArgumentParser
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

DATA_FILENAME_PATTERN = re.compile(r"pubmed22n(?P<file_id>\d+)\.txt")  # r"batch_(?P<batch_id>\d+)\.pt")


def create_dataset_offsets_grouped_by_filename(input_data_dir: str, num_train_samples: Optional[int],
                                               num_val_samples: Optional[int], debug: bool):
    train_offsets: List[int] = []
    train_offset_lowerbounds: List[int] = []
    train_offset_upperbounds: List[int] = []
    train_offsets_filenames: List[str] = []
    val_offsets, val_offset_lowerbounds, val_offset_upperbounds, val_offsets_filenames = None, None, None, None
    num_files = len(os.listdir(input_data_dir))
    # Calculate number of samples
    total_num_samples = 0
    train_indices = None
    val_indices = None
    if num_val_samples is not None or num_train_samples is not None:
        for fname in tqdm(sorted(os.listdir(input_data_dir)), total=num_files):
            print(fname)
            m = re.fullmatch(DATA_FILENAME_PATTERN, fname)
            if m:
                input_fpath = os.path.join(input_data_dir, fname)
                with open(input_fpath, 'r', encoding="utf-8") as inp_file:
                    # Go through dataset and create index
                    for line in inp_file:
                        total_num_samples += 1
        if num_val_samples is not None:
            # Select random line ids for validation
            val_indices = set(random.sample(range(total_num_samples), num_val_samples))
            val_offsets: List[int] = []
            val_offset_lowerbounds: List[int] = []
            val_offset_upperbounds: List[int] = []
            val_offsets_filenames: List[str] = []
        if num_train_samples is not None:
            train_indices = set(random.sample(range(total_num_samples), num_train_samples))
    last_append = None
    line_id = 0
    for fname in tqdm(sorted(os.listdir(input_data_dir)), total=num_files):
        m = re.fullmatch(DATA_FILENAME_PATTERN, fname)
        if m:
            input_fpath = os.path.join(input_data_dir, fname)
            # Updating lowerbounds

            train_offset_lowerbounds.append(len(train_offsets))
            train_offsets_filenames.append(fname)
            if val_indices is not None:
                val_offset_lowerbounds.append(len(val_offsets))
                val_offsets_filenames.append(fname)
            with open(input_fpath, 'r', encoding="utf-8") as inp_file:
                if val_indices is not None and line_id in val_indices:
                    val_offsets.append(inp_file.tell())
                    last_append = "dev"
                else:
                    if train_indices is not None:
                        if line_id in train_indices:
                            train_offsets.append(inp_file.tell())
                            last_append = "train"
                    else:
                        train_offsets.append(inp_file.tell())
                        last_append = "train"
                line_id += 1
                while inp_file.readline():
                    if val_indices is not None and line_id in val_indices:
                        val_offsets.append(inp_file.tell())
                        last_append = "dev"
                    else:
                        if train_indices is not None:
                            if line_id in train_indices:
                                train_offsets.append(inp_file.tell())
                                last_append = "train"
                        else:
                            train_offsets.append(inp_file.tell())
                            last_append = "train"
                    line_id += 1
                line_id -= 1
                if last_append == "train":
                    train_offsets.pop()
                elif last_append == "dev":
                    val_offsets.pop()
                else:
                    raise ValueError(f"last_append: {last_append}")

            train_offset_upperbounds.append(len(train_offsets))
            if val_indices is not None:
                val_offset_upperbounds.append(len(val_offsets))

    # for fname in tqdm(os.listdir(input_data_dir), total=num_files):
    #     m = re.fullmatch(DATA_FILENAME_PATTERN, fname)
    #     if m:
    #         input_fpath = os.path.join(input_data_dir, fname)
    #         offset_lowerbounds.append(len(all_offsets))
    #         offsets_filenames.append(fname)
    #         with open(input_fpath, 'r', encoding="utf-8") as inp_file:
    #             all_offsets.append(inp_file.tell())
    #             while inp_file.readline():
    #                 all_offsets.append(inp_file.tell())
    #             all_offsets.pop()
    #         offset_upperbounds.append(len(all_offsets))
    #
    #     else:
    #         continue
    if debug:
        line_counter = 0
        for fname in tqdm(os.listdir(input_data_dir), total=num_files):
            m = re.fullmatch(DATA_FILENAME_PATTERN, fname)
            if m:
                input_fpath = os.path.join(input_data_dir, fname)

                with open(input_fpath, 'r', encoding="utf-8") as inp_file:
                    for line in inp_file:
                        line_counter += 1
            else:
                continue

    if val_indices is not None:
        print("len(val_offsets)", len(val_offsets))
        print("num_val_samples", num_val_samples)
        # assert len(train_offsets) + len(val_offsets) == total_num_samples
        assert len(val_offsets) == num_val_samples
    else:
        if debug:
            assert len(train_offsets) == line_counter
    train_split = (train_offsets, train_offset_lowerbounds, train_offset_upperbounds, train_offsets_filenames)
    val_split = None
    if val_offsets is not None:
        val_split = (val_offsets, val_offset_lowerbounds, val_offset_upperbounds, val_offsets_filenames)
    print("len(train_offsets)", len(train_offsets))
    print("num_train_samples", num_train_samples)

    return train_split, val_split


def main(args):
    input_data_dir = args.input_data_dir
    num_train_samples = args.num_train_samples
    num_val_samples = args.num_val_samples
    debug = args.debug

    train_split, val_split = create_dataset_offsets_grouped_by_filename(
        input_data_dir,
        num_train_samples=num_train_samples,
        num_val_samples=num_val_samples,
        debug=debug)

    tr_offsets, tr_offset_lowerbounds, tr_offset_upperbounds, tr_offset_filenames = train_split
    if val_split is not None:
        val_offsets, val_offset_lowerbounds, val_offset_upperbounds, val_offset_filenames = val_split
        assert len(val_offset_lowerbounds) == len(val_offset_filenames) == len(val_offset_upperbounds)
        print("val_offsets", len(val_offsets))
        val_offsets = np.array(val_offsets)
        val_offset_lowerbounds = np.array(val_offset_lowerbounds)
        val_offset_upperbounds = np.array(val_offset_upperbounds)

        val_output_offsets_path = os.path.join(input_data_dir, "val_offsets.npy")
        val_output_offset_lowerbounds_path = os.path.join(input_data_dir, "val_offset_lowerbounds.npy")
        val_output_offset_upperbounds_path = os.path.join(input_data_dir, "val_offset_upperbounds.npy")
        val_output_offset_filenames_path = os.path.join(input_data_dir, "val_offset_filenames.txt")

        logging.info(f"Validation:")
        logging.info(f"\tOffsets size: {val_offsets.shape}")
        logging.info(f"\tOffset lowerbounds size: {val_offset_lowerbounds.shape}")
        logging.info(f"\tOffset upperbounds size: {val_offset_upperbounds.shape}")
        logging.info(f"\tOffset filenames length: {len(val_offset_filenames)}")

        np.save(val_output_offsets_path, val_offsets)
        np.save(val_output_offset_lowerbounds_path, val_offset_lowerbounds)
        np.save(val_output_offset_upperbounds_path, val_offset_upperbounds)
        with open(val_output_offset_filenames_path, "w", encoding="utf-8") as out_file:
            s = '\t'.join(val_offset_filenames)
            out_file.write(f"{s}\n")

    assert len(tr_offset_lowerbounds) == len(tr_offset_filenames) == len(tr_offset_upperbounds)
    tr_offsets = np.array(tr_offsets)
    tr_offset_lowerbounds = np.array(tr_offset_lowerbounds)
    tr_offset_upperbounds = np.array(tr_offset_upperbounds)

    tr_output_offsets_path = os.path.join(input_data_dir, "train_offsets.npy")
    tr_output_offset_lowerbounds_path = os.path.join(input_data_dir, "train_offset_lowerbounds.npy")
    tr_output_offset_upperbounds_path = os.path.join(input_data_dir, "train_offset_upperbounds.npy")
    tr_output_offset_filenames_path = os.path.join(input_data_dir, "train_offset_filenames.txt")

    logging.info(f"Train:")
    logging.info(f"\tOffsets size: {tr_offsets.shape}")
    logging.info(f"\tOffset lowerbounds size: {tr_offset_lowerbounds.shape}")
    logging.info(f"\tOffset upperbounds size: {tr_offset_upperbounds.shape}")
    logging.info(f"\tOffset filenames length: {len(tr_offset_filenames)}")

    np.save(tr_output_offsets_path, tr_offsets)
    np.save(tr_output_offset_lowerbounds_path, tr_offset_lowerbounds)
    np.save(tr_output_offset_upperbounds_path, tr_offset_upperbounds)
    with open(tr_output_offset_filenames_path, "w", encoding="utf-8") as out_file:
        s = '\t'.join(tr_offset_filenames)
        out_file.write(f"{s}\n")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--input_data_dir')
    parser.add_argument('--num_train_samples', type=int, required=False)
    parser.add_argument('--num_val_samples', type=int, required=False)
    parser.add_argument('--debug', action="store_true")
    # parser.add_argument('--input_data_dir', required=False,
    #                     default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/2024_feb_debug_dataset/tokenized_sentences")
    # parser.add_argument('--num_val_samples', type=int, required=False, default=100000)
    # parser.add_argument('--debug', required=False, default=True)
    args = parser.parse_args()
    main(args)
