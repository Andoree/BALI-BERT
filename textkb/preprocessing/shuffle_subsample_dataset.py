import logging
import os
import random
import re
import shutil
from argparse import ArgumentParser

from tqdm import tqdm

from textkb.utils.io import create_dir_if_not_exists

BATCH_FILENAME_PATTERN = re.compile(r"batch_(?P<batch_id>\d+)\.pt")


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    create_dir_if_not_exists(output_dir)
    num_samples = args.num_samples
    subsample_ratio = args.subsample_ratio
    if (num_samples is None) == (subsample_ratio is None):
        raise AttributeError(f"Only one option supported: either num_samples or subsample_ratio."
                             f"{num_samples} and {subsample_ratio} were given.")

    # input_fnames = list(os.listdir(input_dir))
    num_batches = 0
    for inp_fname in os.listdir(input_dir):
        m = re.fullmatch(BATCH_FILENAME_PATTERN, inp_fname)
        if m is not None:
            num_batches += 1
        else:
            assert "config" in inp_fname
    if subsample_ratio is not None:
        num_samples = int(num_batches * subsample_ratio)
    sampled_batch_ids = set(random.sample(range(num_batches), num_samples))
    logging.info(f"There are {num_batches} batches. Sampled {len(sampled_batch_ids)} of them.")
    # for i, fname in enumerate(subsampled_fnames):
    i = 0
    logging.info(f"Copying data...")
    for inp_fname in tqdm(os.listdir(input_dir)):
        assert re.fullmatch(BATCH_FILENAME_PATTERN, inp_fname) or "config" in inp_fname
        m = re.fullmatch(BATCH_FILENAME_PATTERN, inp_fname)
        if m is not None:
            batch_id = int(m.group("batch_id"))
            if batch_id in sampled_batch_ids:
                out_fname = BATCH_FILENAME_PATTERN.sub(r"batch_\g<batch_id>.pt", fr"batch_{i}.pt")
                input_path = os.path.join(input_dir, inp_fname)
                output_path = os.path.join(output_dir, out_fname)
                shutil.copyfile(input_path, output_path)
                i += 1


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--input_dir',
                        default="/home/c204/University/NLP/text_kb/textkb/textkb/preprocessing/DELETE/train/")
    parser.add_argument('--num_samples', required=False, default=50, type=int)
    parser.add_argument('--subsample_ratio', required=False, type=int)
    parser.add_argument('--output_dir',
                        default="/home/c204/University/NLP/text_kb/textkb/textkb/preprocessing/DELETE/train_subsample/")
    args = parser.parse_args()
    main(args)
