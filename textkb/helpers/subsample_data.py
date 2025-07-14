import logging
import os
import random
from argparse import ArgumentParser

from textkb.utils.io import create_dir_if_not_exists


def save_files(inp_dir, fnames, out_dir):
    for fname in fnames:
        inp_path = os.path.join(inp_dir, fname)
        out_path = os.path.join(out_dir, fname)
        with open(inp_path, 'r', encoding="utf-8") as inp_file, open(out_path, 'w+', encoding="utf-8") as out_file:
            out_file.write(inp_file.read())


def main(args):
    tokenized_sentences_dir = args.tokenized_sentences_dir
    output_dir = args.output_dir
    output_train_dir = os.path.join(output_dir, "train/")
    output_val_dir = os.path.join(output_dir, "val/")
    create_dir_if_not_exists(output_train_dir)
    create_dir_if_not_exists(output_val_dir)

    fnames = list(os.listdir(tokenized_sentences_dir))
    random.shuffle(fnames)
    num_files = len(fnames)
    num_val_files = args.num_val_files

    if args.num_train_files is None:
        num_train_files = num_files - num_val_files
    else:
        num_train_files = args.num_train_files
    val_fnames = fnames[:num_val_files]
    train_fnames = fnames[num_val_files:num_val_files + num_train_files]

    save_files(tokenized_sentences_dir, train_fnames, output_train_dir)
    save_files(tokenized_sentences_dir, val_fnames, output_val_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument("--tokenized_sentences_dir", type=str,
                        default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/graph_dataset_debug/tokenized_sentences")
    parser.add_argument("--num_train_files", type=int)
    parser.add_argument("--num_val_files", type=int, default=20)
    parser.add_argument("--output_dir", type=str,
                        default="")

    args = parser.parse_args()
    main(args)
