
import logging
import os
from argparse import ArgumentParser

from textkb.utils.io import read_mrconso


def main(args):
    data_cuis = set()
    sep = '\t'
    for fname in os.listdir(args.input_text_dataset_dir):
        logging.info(f"Processing {fname}")
        fpath = os.path.join(args.input_text_dataset_dir, fname)
        with open(fpath, 'r', encoding="utf-8") as inp_file:
            for line in inp_file:
                attrs = line.strip().split(sep)
                cui = attrs[2]
                data_cuis.add(cui)
    logging.info(f"Num unique CUIS in data: {len(data_cuis)}")
    mrconso_df = read_mrconso(args.mrconso)
    mrconso_df = mrconso_df[mrconso_df["CUI"].isin(data_cuis)]
    mrconso_df.to_csv(args.output_mrconso, sep='|', encoding='utf-8', quoting=3, index=False)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )

    parser = ArgumentParser()
    parser.add_argument('--input_text_dataset_dir',
                        default="/home/c204/University/NLP/BERN2_sample/bert_reformated_to_umls/")
    parser.add_argument('--mrconso',
                        default="/home/c204/University/NLP/UMLS/2020AB/ENG_MRCONSO_FILTERED.RRF")
    parser.add_argument('--output_mrconso',
                        default="/home/c204/University/NLP/UMLS/2020AB/ENG_MRCONSO_FILTERED_BY_DATA.RRF")
    args = parser.parse_args()
    main(args)
