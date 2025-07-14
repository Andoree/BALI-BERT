
import logging
import os
from argparse import ArgumentParser


def main(args):
    k = 100
    longest_entities = []
    sep = '\t'
    for fname in os.listdir(args.input_text_dataset_dir):
        logging.info(f"Processing {fname}")
        fpath = os.path.join(args.input_text_dataset_dir, fname)
        with open(fpath, 'r', encoding="utf-8") as inp_file:
            for line in inp_file:
                attrs = line.strip().split(sep)
                if len(attrs) < 3:
                    break
                mention = attrs[3]
                pubmed_id = attrs[0]
                length = len(mention)
                longest_entities.append((length, mention, fname, pubmed_id))

        logging.info("Sorting...")
        longest_entities.sort(key=lambda tt: -tt[0])
        longest_entities = longest_entities[:k]
    for t in longest_entities:
        mention = t[1]
        fname = t[2]
        pubmed_id = t[3]
        print(f"{len(mention)} : {pubmed_id} {fname} {mention}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )

    parser = ArgumentParser()
    parser.add_argument('--input_text_dataset_dir',
                        default="/home/c204/University/NLP/BERN2_sample/bert_reformated/annotations/")
    args = parser.parse_args()
    main(args)
