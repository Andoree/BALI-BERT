import logging
import os
from argparse import ArgumentParser
from typing import List, Tuple, Set, Iterable

from tqdm import tqdm
from transformers import AutoTokenizer
# from nltk import word_tokenize
from nltk.tokenize import TreebankWordTokenizer
from textkb.data.entity import Entity
from textkb.utils.io import create_dir_if_not_exists, load_entities_groupby_doc_id_sent_id, load_dict, save_dict, \
    load_node_id2terms_list, save_node_id2terms_list
from textkb.utils.tokenization import create_sentences_word_vocab_nltk, transformer_vocab2input_ids, \
    transformer_tokenize_str_list


def main(args):
    output_path = args.output_path
    output_dir = os.path.dirname(output_path)
    create_dir_if_not_exists(output_dir)

    args_fname = os.path.join(output_dir, "concept_tokenization_config.txt")
    save_dict(save_path=args_fname, dictionary=vars(args), )

    transformer_tokenizer = AutoTokenizer.from_pretrained(args.transformer_tokenizer_name,
                                                          do_lower_case=args.do_lower_case)

    node_id2terms = load_node_id2terms_list(args.input_node2terms_path)
    if args.do_lower_case:
        node_id2terms = [[x.lower() for x in t] for t in node_id2terms]

    unique_terms = set()
    logging.info("Creating set of unique terms")
    for terms_list in node_id2terms:
        unique_terms.update(terms_list)
    unique_terms = list(unique_terms)
    logging.info(f"There are {len(unique_terms)} unique terms. Tokenizing....")
    max_length = args.max_length

    input_ids = transformer_tokenize_str_list(string_list=unique_terms,
                                              transformer_tokenizer=transformer_tokenizer,
                                              add_special_tokens=True,
                                              truncation=True, max_length=max_length,
                                              return_attention_mask=False,
                                              return_token_type_ids=False)

    assert len(input_ids) == len(unique_terms)
    term2inp_ids = {term: inp_ids for term, inp_ids in zip(unique_terms, input_ids)}

    node_id2input_ids = [map(lambda x: term2inp_ids[x], terms_list) for terms_list in node_id2terms]
    with open(output_path, 'w+', encoding="utf-8") as out_file:
        for inp_ids_list in node_id2input_ids:
            s = "|||".join(map(lambda t: ','.join(str(x) for x in t[:max_length]), inp_ids_list))
            out_file.write(f"{s}\n")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--input_node2terms_path', )
    parser.add_argument('--transformer_tokenizer_name')
    parser.add_argument('--do_lower_case', action="store_true")
    parser.add_argument('--output_path', )
    parser.add_argument('--max_length', type=int)

    # parser.add_argument('--input_node2terms_path',
    #                     default="/home/c204/University/NLP/BERN2_sample/debug_graph/node_id2terms_list")  # TODO!!
    # parser.add_argument('--transformer_tokenizer_name', default="prajjwal1/bert-tiny")
    # parser.add_argument('--do_lower_case', default=True)
    # parser.add_argument('--output_path',
    #                     default="/home/c204/University/NLP/BERN2_sample/debug_graph/node_id2terms_list_tinybert_test")
    # parser.add_argument('--max_length', type=int, default=48)
    args = parser.parse_args()
    main(args)
