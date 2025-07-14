import os
import random
from typing import Dict, Tuple, List, Union

import numpy as np
import torch

from textkb.utils.io import load_dict


def get_list_min_max_length(lst, key_f=lambda x: x):
    min_val = None
    max_val = None
    length = 0
    for elem in lst:
        elem_val = key_f(elem)
        if min_val is None and max_val is None:
            min_val, max_val = elem_val, elem_val
        min_val = min(min_val, elem_val)
        max_val = max(max_val, elem_val)

        length += 1
    return min_val, max_val, length


def validate_sentence_concept_tokenization(inp_tok_sentences_dir: str, tokenized_concepts_path: str):
    tokenized_sent_config_path = os.path.join(inp_tok_sentences_dir, "config.txt")
    sent_tok_config = load_dict(tokenized_sent_config_path, sep='\t')
    sent_tok_do_lower_case = sent_tok_config["do_lower_case"]
    sent_tok_model_name = sent_tok_config["transformer_tokenizer_name"]

    tokenized_concepts_dir = os.path.dirname(tokenized_concepts_path)
    tokenized_concept_config_path = os.path.join(tokenized_concepts_dir, "concept_tokenization_config.txt")
    concept_tok_config = load_dict(tokenized_concept_config_path, sep='\t')
    concept_tok_do_lower_case = concept_tok_config["do_lower_case"]
    concept_tok_model_name = concept_tok_config["transformer_tokenizer_name"]

    assert concept_tok_do_lower_case == sent_tok_do_lower_case
    assert sent_tok_model_name == concept_tok_model_name


def create_t2hr_adjacency_lists_from_h2rt(h2rt_adjacency_lists: Dict[int, Tuple[Union[Tuple[int, int, int],
Tuple[int]]]]):
    t2hr_adjacency_lists: Dict[int, List[Tuple[int, int]]] = {}
    for h, rt_list in h2rt_adjacency_lists.items():
        for edge in rt_list:
            t = edge[0]
            r = edge[1]
            if t2hr_adjacency_lists.get(t) is None:
                t2hr_adjacency_lists[t] = []
            t2hr_adjacency_lists[t].append((h, r))
    return t2hr_adjacency_lists


def token_ids2str(tokenizer, token_ids, spec_token_ids):
    # mask_token_id = tokenizer.mask_token_id
    # cls_token_id = tokenizer.cls_token_id
    # sep_token_id = tokenizer.sep_token_id
    # pad_token_id = tokenizer.pad_token_id

    tokens = tokenizer.convert_ids_to_tokens([x for x in token_ids if x not in spec_token_ids])
    s = "".join((x.strip("#") if x.startswith("#") else f" {x}" for x in tokens))

    return s


def convert_input_ids_list_to_str_list(input_ids_list, tokenizer, spec_token_ids):
    str_list = [tokenizer.convert_ids_to_tokens([x for x in t if x not in spec_token_ids]) for t in
                input_ids_list]
    str_list = ["".join((x.strip("#") if x.startswith("#") else f" {x}" for x in t)) for t in
                str_list]

    return str_list


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.random.manual_seed(seed)
    torch.cuda.random.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
