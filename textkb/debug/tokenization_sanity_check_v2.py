# TODO: Попробовать развернуть токенизированные тексты обратно, притануть рёбра


import logging
import os
import random
from argparse import ArgumentParser

from transformers import AutoTokenizer

from textkb.utils.io import read_mrconso, load_dict, load_node_id2terms_list, load_list_elem_per_line, \
    load_tokenized_concepts


def main(args):
    transformer_tokenizer = AutoTokenizer.from_pretrained(args.transformer_tokenizer_name)
    # node_id2cui = load_dict(args.node_id2cui_path, dtype_1=int, dtype_2=str)
    node_id2terms_path = args.node_id2terms_path
    node_id2tokenized_terms = load_tokenized_concepts(node_id2terms_path)
    graph_data_dir = os.path.dirname(node_id2terms_path)
    mentioned_concepts_idx_path = os.path.join(graph_data_dir, "mentioned_concepts_idx")
    local2global_concept_id = load_list_elem_per_line(input_path=mentioned_concepts_idx_path, dtype=int)
    global2local_concept_id = {global_id: local_id for local_id, global_id in enumerate(local2global_concept_id)}

    # TODO: КАК CUI-LESS ОКАЗАЛСЯ ВО ВХОДЕ?
    sep = '\t'
    for fname in os.listdir(args.tokenized_data_dir):
        if fname == "config.txt":
            continue
        logging.info(f"Processing {fname}")
        fpath = os.path.join(args.tokenized_data_dir, fname)
        with open(fpath, 'r', encoding="utf-8") as inp_file:  # encoding="ascii") as inp_file:
            print(fname)
            for line in inp_file:
                attrs = line.strip().split(sep)
                data = tuple(map(int, line.strip().split(',')))
                inp_ids_end, token_mask_end, ei_tok_idx_end, ei_ent_idx_end = data[:4]

                input_ids = data[4:inp_ids_end + 4]
                token_ent_binary_mask = data[inp_ids_end + 4:token_mask_end + 4]
                edge_index_token_idx = data[token_mask_end + 4:ei_tok_idx_end + 4]
                edge_index_entity_idx = data[ei_tok_idx_end + 4:ei_ent_idx_end + 4]

                # pubmed_id_sent_id = attrs[0]
                # input_ids = list(int(x) for x in attrs[1].split(','))
                # token_ent_binary_mask = list(int(x) for x in attrs[2].split(','))
                # edge_index_token_idx = list(int(x) for x in attrs[3].split(','))
                # edge_index_entity_idx = list(int(x) for x in attrs[4].split(','))

                sent_tokens = transformer_tokenizer.convert_ids_to_tokens(input_ids)
                sentence = "".join((x.strip("#") if x.startswith("#") else f" {x}" for x in sent_tokens))

                assert len(input_ids) == len(token_ent_binary_mask)
                # print(len(input_ids), "input_ids", input_ids)
                print(len(sent_tokens), "sent_tokens", sent_tokens)
                print("SENTENCE", sentence)

                # print(len(token_ent_binary_mask), "token_ent_binary_mask", sum(token_ent_binary_mask),
                #       token_ent_binary_mask)
                # print(len(edge_index_token_idx), "edge_index_token_idx", edge_index_token_idx)
                # print(len(edge_index_entity_idx), "edge_index_entity_idx", edge_index_entity_idx)
                # print('--' * 10)
                sent_tokens = [token for m, token in zip(token_ent_binary_mask, sent_tokens) if m == 1]
                sampled_tokens = [sent_tokens[token_id] for token_id in edge_index_token_idx]
                # TODO "".join((x.strip("#") if x.startswith("#") else f" {x}" for x in t
                sampled_node_terms = [transformer_tokenizer.convert_ids_to_tokens(node_id2tokenized_terms[node_id][0])
                                      for node_id in edge_index_entity_idx]
                sampled_node_terms = ["".join((x.strip("#") if x.startswith("#") else f" {x}" for x in t)) for t in
                                      sampled_node_terms]
                assert len(sampled_tokens) == len(sampled_node_terms)
                # TODO: UNCOMMENT!!!
                for token, term in zip(sampled_tokens, sampled_node_terms):
                    print(f"{token} ---> {term}")
                print('---')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )

    parser = ArgumentParser()
    # parser.add_argument('--tokenized_data_dir',
    #                     default="/home/c204/University/NLP/BERN2_sample/TOKENIZED_UNMASKED")
    # parser.add_argument('--node_id2cui_path',
    #                     default="/home/c204/University/NLP/BERN2_sample/debug_graph/id2cui")
    # parser.add_argument('--transformer_tokenizer_name', default="prajjwal1/bert-tiny")
    # parser.add_argument('--node_id2terms_path',
    #                     default="/home/c204/University/NLP/BERN2_sample/debug_graph/node_id2terms_list")
    parser.add_argument('--tokenized_data_dir',
                        default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/graph_dataset_debug/v2_tokenized_sentences")
    # parser.add_argument('--node_id2cui_path',
    #                     default="/home/c204/University/NLP/BERN2_sample/debug_graph/id2cui")
    parser.add_argument('--transformer_tokenizer_name',
                        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    # default="prajjwal1/bert-tiny")
    parser.add_argument('--node_id2terms_path',
                        default="/home/c204/University/NLP/BERN2_sample/graph_dataset_debug/graph_dataset_debug/node_id2terms_list_tokenized_BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

    args = parser.parse_args()
    main(args)
