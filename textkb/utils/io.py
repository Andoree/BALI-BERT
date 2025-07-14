import logging
import multiprocessing
import os
from multiprocessing import Pool
from typing import Dict, Set, List, Tuple, Optional, Union, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from textkb.data.entity import Entity


def create_dir_if_not_exists(output_dir):
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)


def load_dict(path: str, sep: str = '\t', dtype_1=str, dtype_2=str, ) -> Dict:
    df = pd.read_csv(path, header=None, names=("key", "value"), sep=sep, encoding="utf-8")
    return dict(zip(map(dtype_1, df.key), map(dtype_2, df.value)))


def read_mrconso(fpath, usecols=None) -> pd.DataFrame:
    columns = ['CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI', 'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE',
               'STR', 'SRL', 'SUPPRESS', 'CVF', 'NOCOL']
    return pd.read_csv(fpath, names=columns, sep='|', encoding='utf-8', quoting=3, usecols=usecols)


def read_mrsty(fpath) -> pd.DataFrame:
    columns = ['CUI', 'TUI', 'STN', 'STY', 'ATUI', 'CVF', 'NOCOL']
    return pd.read_csv(fpath, names=columns, sep='|', encoding='utf-8', quoting=3)


def read_mrrel(fpath, usecols=None) -> pd.DataFrame:
    columns = ["CUI1", "AUI1", "STYPE1", "REL", "CUI2", "AUI2", "STYPE2", "RELA", "RUI", "SRUI", "SAB",
               # "RSAB", "VSAB",
               "SL", "RG", "DIR", "SUPPRESS", "CVF", 'NOCOL']
    return pd.read_csv(fpath, names=columns, sep='|', encoding='utf-8', usecols=usecols)


def read_mrdef(fpath) -> pd.DataFrame:
    columns = ["CUI", "AUI", "ATUI", "SATUI", "SAB", "DEF", "SUPPRESS", "CVF", 'NOCOL']
    return pd.read_csv(fpath, names=columns, sep='|', encoding='utf-8')


def get_text_dataset_unique_cuis_max_mention_length(input_dir: str, sep='\t'):
    unique_cuis: Set[str] = set()
    max_mention_length = 0
    logging.info(f"Getting unique cuis set. Processing data files")
    for fname in tqdm(os.listdir(input_dir), total=len(os.listdir(input_dir))):
        inp_path = os.path.join(input_dir, fname)
        with open(inp_path, 'r', encoding="utf-8") as inp_file:
            for line in inp_file:
                attrs = line.strip().split(sep)
                cui = attrs[2]
                if cui in ("CUI-less", "CUILESS"):
                    continue
                mention = attrs[4]
                max_mention_length = max(max_mention_length, len(mention))
                unique_cuis.add(cui)
    logging.info(f"# Unique cuis: {len(unique_cuis)}. Max mention length: {max_mention_length}")
    return unique_cuis, max_mention_length


def save_dict(save_path: str, dictionary: Dict, sep: str = '\t'):
    logging.info("Saving dictionary")
    with open(save_path, 'w+', encoding="utf-8") as out_file:
        for key, val in dictionary.items():
            key, val = str(key), str(val)
            if sep in key or sep in val:
                raise Exception(f"Separator {sep} is present in dictionary being saved")
            out_file.write(f"{key}{sep}{val}\n")
    logging.info("Finished saving dictionary")


def save_tuples(save_path: str, tuples: List[Tuple], sep='\t'):
    logging.info("Saving tuples")
    with open(save_path, 'w+', encoding="utf-8") as out_file:
        for t in tuples:
            s = sep.join((str(x) for x in t))
            out_file.write(f"{s}\n")
    logging.info("Finished saving tuples")


def save_node_id2terms_list(save_path: str, node_id2terms: Dict[int, List[str]], node_terms_sep: str = '\t',
                            terms_sep: str = '|||', drop_keys=False):
    num_concepts = len(node_id2terms)
    logging.info(f"Saving CUIs and terms. There are {num_concepts}")
    with open(save_path, 'w+', encoding="utf-8") as out_file:
        for i, node_id in tqdm(enumerate(sorted(node_id2terms.keys())), miniters=num_concepts // 50,
                               total=num_concepts):
            assert i == node_id
            terms_list = node_id2terms[node_id]

            assert node_terms_sep not in str(node_id) and terms_sep not in str(node_id)
            for term in terms_list:
                if node_terms_sep in term:
                    raise ValueError(f"col_sep {node_terms_sep} is present in data being saved")
                if terms_sep in term:
                    raise ValueError(f"terms_sep {terms_sep} is present in data being saved")
            terms_str = terms_sep.join(terms_list)
            if drop_keys:
                out_file.write(f"{terms_str}\n")
            else:
                out_file.write(f"{node_id}{node_terms_sep}{terms_str}\n")
    logging.info("Finished saving CUIs and terms")


def save_adjacency_lists(save_path: str, adjacency_lists: Dict[int, List[Tuple[int, int, int]]],
                         node_adj_list_sep: str = '\t', edge_sep='|', neighbor_id_sep: str = ',', ):
    num_central_concepts = len(adjacency_lists.keys())
    logging.info(f"Saving adjacency lists. {num_central_concepts} nodes have at least one neighbor")
    with open(save_path, 'w+', encoding="utf-8") as out_file:
        for node_id, adj_list in tqdm(adjacency_lists.items(), total=num_central_concepts,
                                      miniters=num_central_concepts // 100):
            s = edge_sep.join(neighbor_id_sep.join(str(x) for x in t) for t in adj_list)
            out_file.write(f"{node_id}{node_adj_list_sep}{s}\n")


def load_entities_groupby_doc_id_sent_id(input_path: str, drop_cuiless: bool, do_lower_case: bool,
                                         cui2node_id: Optional[Dict[str, int]], field_sep="||", subfield_sep='|', ) \
        -> Dict[int, Dict[int, List[Entity]]]:
    doc_sent_id2entities = {}
    with open(input_path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            attrs = line.strip().split(field_sep)
            doc_id = int(attrs[0])
            sentence_id = int(attrs[1])
            if doc_sent_id2entities.get(doc_id) is None:
                doc_sent_id2entities[doc_id] = {}
            if doc_sent_id2entities[doc_id].get(sentence_id) is None:
                doc_sent_id2entities[doc_id][sentence_id] = []

            concept_ids_str = attrs[2]
            if drop_cuiless and concept_ids_str in ("CUI-less", "CUILESS"):
                continue
            concept_ids = concept_ids_str.split(subfield_sep)
            if cui2node_id is not None:
                if len(concept_ids) == 1 and concept_ids[0] in ("CUI-less", "CUILESS"):
                    continue

                concept_ids = tuple(
                    filter(lambda y: y is not None,
                           map(lambda x: cui2node_id.get(x, None), concept_ids))
                )

            mention = attrs[3]
            if do_lower_case:
                mention = mention.lower()
            span = attrs[5]
            span_start, span_end = (int(x) for x in span.split(subfield_sep))

            entity = Entity(mention_str=mention, span_start=span_start, span_end=span_end, node_ids=concept_ids)
            doc_sent_id2entities[doc_id][sentence_id].append(entity)
    return doc_sent_id2entities


def load_node_id2terms_list(input_path: str, terms_sep: str = '|||'):
    node_id2terms: List[List[str]] = []

    with open(input_path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            terms_list = line.strip().split(terms_sep)
            node_id2terms.append(terms_list)
    return node_id2terms


def load_tokenized_sentences_file(tokenized_data_dir, fname, field_sep, input_ids_list,
                                  token_entity_mask_list, edge_index_token_idx_list, edge_index_entity_idx_list):
    if fname == "config.txt":
        pass
    else:
        fpath = os.path.join(tokenized_data_dir, fname)
        with open(fpath, 'r', encoding="utf-8") as inp_file:
            for line in inp_file:
                attrs = line.strip().split(field_sep)

                inp_ids = tuple(int(x) for x in attrs[1].split(','))
                token_mask = tuple(int(x) for x in attrs[2].split(','))
                edge_index_token_idx = tuple(int(x) for x in attrs[3].split(','))
                edge_index_entity_idx = tuple(int(x) for x in attrs[4].split(','))

                assert len(inp_ids) == len(token_mask)
                assert len(edge_index_token_idx) == len(edge_index_entity_idx)

                input_ids_list.append(inp_ids)
                token_entity_mask_list.append(token_mask)
                edge_index_token_idx_list.append(edge_index_token_idx)
                edge_index_entity_idx_list.append(edge_index_entity_idx)


def loading_decorator(tokenized_data_dir, field_sep, input_ids_list,
                      token_entity_mask_list, edge_index_token_idx_list, edge_index_entity_idx_list):
    def f(fname):
        return load_tokenized_sentences_file(tokenized_data_dir, fname, field_sep, input_ids_list,
                                             token_entity_mask_list, edge_index_token_idx_list,
                                             edge_index_entity_idx_list)

    return f


# def load_tokenized_sentences_data_parallel(tokenized_data_dir: str, n_proc, field_sep: str = '\t'):
#     pool = Pool(processes=n_proc)
#     # input_ids_list: List[Tuple[int]] = []
#     # token_entity_mask_list: List[Tuple[int]] = []
#     # edge_index_token_idx_list: List[Tuple[int]] = []
#     # edge_index_entity_idx_list: List[Tuple[int]] = []
#     input_ids_list: List[Tuple[int]] = []
#     token_entity_mask_list: List[Tuple[int]] = []
#     edge_index_token_idx_list: List[Tuple[int]] = []
#     edge_index_entity_idx_list: List[Tuple[int]] = []
#     with multiprocessing.Manager() as manager:
#         manager.list()
#         input_ids_list: List[Tuple[int]] = manager.list()
#         token_entity_mask_list: List[Tuple[int]] = manager.list()
#         edge_index_token_idx_list: List[Tuple[int]] = manager.list()
#         edge_index_entity_idx_list: List[Tuple[int]] = manager.list()
#         logging.info(f"Loading tokenized sentences from: {tokenized_data_dir}")
#         fnames = list(os.listdir(tokenized_data_dir))
#         inputs = [(tokenized_data_dir, x, field_sep, input_ids_list,
#                    token_entity_mask_list, edge_index_token_idx_list,
#                    edge_index_entity_idx_list) for x in fnames]
#         # pool.starmap(lambda x: load_tokenized_sentences_file(tokenized_data_dir, x, field_sep, input_ids_list,
#         #                                                  token_entity_mask_list, edge_index_token_idx_list,
#         #                                                  edge_index_entity_idx_list), fnames)
#         pool.starmap(load_tokenized_sentences_file, inputs)
#         # num_files = len(list(os.listdir(tokenized_data_dir)))
#         # for fname in tqdm(os.listdir(tokenized_data_dir), total=num_files, mininterval=10.0):
#         #     load_tokenized_sentences_file(tokenized_data_dir, fname, field_sep, input_ids_list,
#         #                                   token_entity_mask_list, edge_index_token_idx_list, edge_index_entity_idx_list)
#         pool.close()
#     logging.info(f"Tokenized sentences are loaded. There are {len(input_ids_list)} sentences.")
#     d = {
#         "input_ids": input_ids_list,
#         "token_entity_mask": token_entity_mask_list,
#         "edge_index_token_idx": edge_index_token_idx_list,
#         "edge_index_entity_idx": edge_index_entity_idx_list
#     }
#     return d

def load_tokenized_sentences_data_v2(tokenized_data_dir: str, field_sep: str = '\t'):
    input_ids_list: List[Tuple[int]] = []
    token_entity_mask_list: List[Tuple[int]] = []
    edge_index_token_idx_list: List[Tuple[int]] = []
    edge_index_entity_idx_list: List[Tuple[int]] = []
    logging.info(f"Loading tokenized sentences from: {tokenized_data_dir}")
    num_files = len(list(os.listdir(tokenized_data_dir)))
    for fname in tqdm(os.listdir(tokenized_data_dir), total=num_files, mininterval=10.0):
        if fname == "config.txt":
            continue
        fpath = os.path.join(tokenized_data_dir, fname)
        with open(fpath, 'r', encoding="utf-8") as inp_file:
            for line in inp_file:
                data = tuple(map(int, line.strip().split(',')))
                inp_ids_end, token_mask_end, ei_tok_idx_end, ei_ent_idx_end = data[:4]
                # data = tuple(map(int, attrs[1].split(',')))
                # print("data", data)
                inp_ids = data[4:inp_ids_end + 4]
                token_mask = data[inp_ids_end + 4:token_mask_end + 4]
                edge_index_token_idx = data[token_mask_end + 4:ei_tok_idx_end + 4]
                edge_index_entity_idx = data[ei_tok_idx_end + 4:ei_ent_idx_end + 4]
                # print("inp_ids", inp_ids)
                # print("token_mask", token_mask)
                # print("edge_index_token_idx", edge_index_token_idx)
                # print("edge_index_entity_idx", edge_index_entity_idx)
                # assert len(inp_ids) == len(token_mask)
                # assert len(inp_ids) + len(token_mask) + len(edge_index_token_idx) + len(edge_index_entity_idx) == len(data)

                input_ids_list.append(inp_ids)
                token_entity_mask_list.append(token_mask)
                edge_index_token_idx_list.append(edge_index_token_idx)
                edge_index_entity_idx_list.append(edge_index_entity_idx)
    logging.info(f"Tokenized sentences are loaded. There are {len(input_ids_list)} sentences.")
    d = {
        "input_ids": input_ids_list,
        "token_entity_mask": token_entity_mask_list,
        "edge_index_token_idx": edge_index_token_idx_list,
        "edge_index_entity_idx": edge_index_entity_idx_list
    }
    return d


def load_tokenized_sentences_data(tokenized_data_dir: str, field_sep: str = '\t'):
    input_ids_list: List[Tuple[int]] = []
    token_entity_mask_list: List[Tuple[int]] = []
    edge_index_token_idx_list: List[Tuple[int]] = []
    edge_index_entity_idx_list: List[Tuple[int]] = []
    logging.info(f"Loading tokenized sentences from: {tokenized_data_dir}")
    num_files = len(list(os.listdir(tokenized_data_dir)))
    for fname in tqdm(os.listdir(tokenized_data_dir), total=num_files, mininterval=10.0):
        if fname == "config.txt":
            continue
        fpath = os.path.join(tokenized_data_dir, fname)
        with open(fpath, 'r', encoding="utf-8") as inp_file:
            for line in inp_file:
                # attrs = line.strip().split(field_sep)

                # inp_ids = tuple(int(x) for x in attrs[1].split(','))
                # token_mask = tuple(int(x) for x in attrs[2].split(','))
                # edge_index_token_idx = tuple(int(x) for x in attrs[3].split(','))
                # edge_index_entity_idx = tuple(int(x) for x in attrs[4].split(','))
                inp_ids, token_mask, edge_index_token_idx, edge_index_entity_idx = (
                    tuple(int(x) for x in s.split(',')) for s in line.strip().split(field_sep)[1:])

                assert len(inp_ids) == len(token_mask)
                assert len(edge_index_token_idx) == len(edge_index_entity_idx)

                input_ids_list.append(inp_ids)
                token_entity_mask_list.append(token_mask)
                edge_index_token_idx_list.append(edge_index_token_idx)
                edge_index_entity_idx_list.append(edge_index_entity_idx)
    logging.info(f"Tokenized sentences are loaded. There are {len(input_ids_list)} sentences.")
    d = {
        "input_ids": input_ids_list,
        "token_entity_mask": token_entity_mask_list,
        "edge_index_token_idx": edge_index_token_idx_list,
        "edge_index_entity_idx": edge_index_entity_idx_list
    }
    return d


def create_edge_pairs(edges_list: Tuple[Tuple[int, int, int]], rel_idx: int) -> Tuple[Tuple[int, int]]:
    new_edges = tuple(map(lambda t: (int(t[0]), int(t[rel_idx])), edges_list))

    return new_edges


def load_adjacency_lists(input_path: str, use_rel: bool, drop_selfloops=False,
                         node_adj_list_sep: str = '\t', edge_sep='|', neighbor_id_sep: str = ',',
                         use_rel_or_rela="rel") \
        -> Dict[int, Tuple[Union[Tuple[int, int], int]]]:
    adjacency_lists: Dict[int, Tuple[Union[Tuple[int, int], int]]] = {}
    num_edges = 0
    num_dropped_sefloops = 0
    logging.info("Loading adjacency lists...")
    with open(input_path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            node_id, neighbors_str = line.strip().split(node_adj_list_sep)
            node_id = int(node_id)
            edge_str_list = neighbors_str.split(edge_sep)
            edges = tuple(set(
                map(tuple, (map(lambda t: map(int, t), (s.split(neighbor_id_sep) for s in edge_str_list))))
            ))

            if drop_selfloops:
                len_before = len(edges)
                edges = tuple(filter(lambda t: False if node_id == t[0] else True, edges))
                len_after = len(edges)
                num_dropped_sefloops += (len_before - len_after)

            if not use_rel:
                edges = tuple(set((t[0] for t in edges)))
            else:
                if use_rel_or_rela == "rel":
                    rel_idx = 1
                elif use_rel_or_rela == "rela":
                    rel_idx = 2
                else:
                    raise ValueError(f"Invalid use_rel_or_rela: {use_rel_or_rela}")
                edges = create_edge_pairs(edges, rel_idx)

                if len(edges) > 0:
                    assert isinstance(edges[0][0], int)

            adjacency_lists[node_id] = edges
            num_edges += len(edges)

    logging.info(f"Adjacency lists are loaded. "
                 f"{len(adjacency_lists)} nodes have a neighbor. "
                 f"# unique edges: {num_edges}. Dropped {num_dropped_sefloops} self-loops")

    return adjacency_lists


def load_tokenized_concepts(tok_conc_path: str, term_sep='|||', id_sep=',') -> List[Tuple[Tuple[int]]]:
    logging.info(f"Loading tokenized concept names from: {tok_conc_path}")
    node_id2tok_concept_names = []
    with open(tok_conc_path, 'r', encoding="utf-8") as inp_file:
        for line in tqdm(inp_file, miniters=100000):
            names_list = line.strip().split(term_sep)
            names_list = tuple((tuple(map(int, s.split(id_sep))) for s in names_list))
            node_id2tok_concept_names.append(names_list)
    logging.info(f"Concept names loaded. There are {len(node_id2tok_concept_names)} concepts")
    return node_id2tok_concept_names


def update_log_file(path: str, dict_to_log: Dict):
    with open(path, 'a+', encoding="utf-8") as out_file:
        s = ', '.join((f"{k} : {v}" for k, v in dict_to_log.items()))
        out_file.write(f"{s}\n")


def save_bert_encoder_tokenizer_config(bert_encoder, bert_tokenizer, save_path: str):
    output_dir = os.path.dirname(save_path)
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)
    logging.info(f"Saving textual encoder and tokenizer to {save_path}")
    bert_encoder.cpu().save_pretrained(save_path)
    bert_encoder.config.save_pretrained(save_path)
    bert_tokenizer.save_pretrained(save_path)
    logging.info(f"Successfully saved textual encoder and tokenizer to {save_path}")


def get_unique_mentioned_concept_ids_from_data(tokenized_data_dir: str):
    unique_mentioned_entity_ids: Set[int] = set()
    logging.info(f"Loading tokenized sentences from: {tokenized_data_dir}")
    num_files = len(list(os.listdir(tokenized_data_dir)))
    for fname in tqdm(os.listdir(tokenized_data_dir), total=num_files, mininterval=10.0):
        if fname == "config.txt":
            continue
        fpath = os.path.join(tokenized_data_dir, fname)
        with open(fpath, 'r', encoding="utf-8") as inp_file:
            for line in inp_file:
                data = tuple(map(int, line.strip().split(',')))
                inp_ids_end, token_mask_end, ei_tok_idx_end, ei_ent_idx_end = data[:4]

                edge_index_entity_idx = data[ei_tok_idx_end + 4:ei_ent_idx_end + 4]
                unique_mentioned_entity_ids.update(edge_index_entity_idx)
    logging.info(f"Found unique mentioned concept ids: {len(unique_mentioned_entity_ids)} unique concepts")

    return unique_mentioned_entity_ids


def save_list_elem_per_line(lst: Iterable, output_path: str):
    with open(output_path, 'w', encoding="utf-8") as out_file:
        for elem in lst:
            out_file.write(f"{elem}\n")


def load_list_elem_per_line(input_path: str, dtype=None) -> Tuple:
    lst = []
    with open(input_path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            elem = line.strip()
            if dtype is not None:
                elem = dtype(elem)
            lst.append(elem)
    lst = tuple(lst)

    return lst


def load_offset_index(input_dir: str, prefix=str):
    offsets_path = os.path.join(input_dir, f"{prefix}_offsets.npy")
    offset_lowerbounds_path = os.path.join(input_dir, f"{prefix}_offset_lowerbounds.npy")
    offset_upperbounds_path = os.path.join(input_dir, f"{prefix}_offset_upperbounds.npy")
    offset_filenames_path = os.path.join(input_dir, f"{prefix}_offset_filenames.txt")
    logging.info(f"Loading offsets index:\n\toffsets: {offsets_path}\n\toffset_lowerbounds: "
                 f"{offset_lowerbounds_path}\n\toffset_filenames: {offset_filenames_path}")

    offsets = np.load(offsets_path)
    offset_lowerbounds = np.load(offset_lowerbounds_path)
    offset_upperbounds = np.load(offset_upperbounds_path)
    with open(offset_filenames_path, 'r', encoding="utf-8") as inp_file:
        offset_filenames = inp_file.readline().strip().split('\t')

    return offsets, offset_lowerbounds, offset_upperbounds, offset_filenames


def untokenize_concept_names(path: str, tokenizer, concept_names_sep: str="|||"):

    concept_names: List[str] = []
    with open(path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            tokenized_concept = line.strip().split(concept_names_sep)[0]
            tokenized_concept_input_ids = [int(x.strip()) for x in tokenized_concept.split(',')]
            tokens = tokenizer.convert_ids_to_tokens(tokenized_concept_input_ids)
            s = "".join((x.strip("#") if x.startswith("#") else f" {x}" for x in tokens))
            concept_names.append(s)

    return concept_names


