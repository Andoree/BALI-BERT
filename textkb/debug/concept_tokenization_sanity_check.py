
import logging
import os
import random
from argparse import ArgumentParser

from transformers import AutoTokenizer

from textkb.utils.io import read_mrconso, load_dict, load_node_id2terms_list


def main(args):
    transformer_tokenizer = AutoTokenizer.from_pretrained(args.transformer_tokenizer_name,
                                                          do_lower_case=True)
    node_id2input_ids = load_node_id2terms_list(args.tokenized_terms_path)

    for i, input_ids_str in enumerate(node_id2input_ids):
        print(input_ids_str)
        print(input_ids_str[0])
        input_ids_str = [int(x) for x in input_ids_str[0].split(',')]
        # print(input_ids_str)
        t = [transformer_tokenizer.convert_ids_to_tokens(input_ids) for input_ids in input_ids_str]
        print(t)
        print('--' * 10)
        if i > 5:
            break

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )

    parser = ArgumentParser()
    parser.add_argument('--tokenized_terms_path',
                        default="/home/c204/University/NLP/BERN2_sample/debug_graph/node_id2terms_list_tinybert_test")
    parser.add_argument('--transformer_tokenizer_name', default="prajjwal1/bert-tiny")
    args = parser.parse_args()
    main(args)
