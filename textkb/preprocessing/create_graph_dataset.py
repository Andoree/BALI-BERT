import logging
import os
from argparse import ArgumentParser
from typing import Dict, List, Tuple

import pandas as pd

from textkb.utils.io import get_text_dataset_unique_cuis_max_mention_length, read_mrrel, read_mrconso, save_dict, \
    save_adjacency_lists, save_node_id2terms_list
from textkb.utils.umls2graph import get_concept_list_groupby_cui, create_graph_adjacency_lists
from textkb.utils.utils import get_list_min_max_length


def create_graph_files(mrconso_df: pd.DataFrame, mrrel_df: pd.DataFrame, rel2id: Dict[str, int],
                       cui2node_id: Dict[str, int], rela2id: Dict[str, int], output_node_id2terms_list_path: str,
                       output_node_id2cui_path: str, output_cui2node_id_path: str,
                       output_adjacency_lists_path: str, output_rel2rel_id_path: str,
                       output_rela2rela_id_path, ignore_not_mapped_edges: bool):
    node_id2cui: Dict[int, str] = {node_id: cui for cui, node_id in cui2node_id.items()}
    node_id2terms_list = get_concept_list_groupby_cui(mrconso_df=mrconso_df, cui2node_id=cui2node_id)
    min_n_id, max_n_id, num_nodes = get_list_min_max_length(node_id2terms_list.keys())
    assert (max_n_id + 1 == num_nodes) and min_n_id == 0
    logging.info("Generating edges....")

    adjacency_lists: Dict[int, List[Tuple[int, int, int]]] = create_graph_adjacency_lists(mrrel_df=mrrel_df,
                                                                                          cui2node_id=cui2node_id,
                                                                                          rel2rel_id=rel2id,
                                                                                          rela2rela_id=rela2id,
                                                                                          ignore_not_mapped_edges=ignore_not_mapped_edges)

    logging.info("Saving the result....")
    save_node_id2terms_list(save_path=output_node_id2terms_list_path, node_id2terms=node_id2terms_list, drop_keys=True)
    save_dict(save_path=output_node_id2cui_path, dictionary=node_id2cui)
    save_dict(save_path=output_cui2node_id_path, dictionary=cui2node_id)
    save_dict(save_path=output_rel2rel_id_path, dictionary=rel2id)
    save_dict(save_path=output_rela2rela_id_path, dictionary=rela2id)
    save_adjacency_lists(save_path=output_adjacency_lists_path, adjacency_lists=adjacency_lists)


def create_cui2node_id_mapping(mrconso_df: pd.DataFrame) -> Dict[str, int]:
    unique_cuis_set = set(mrconso_df["CUI"].unique())
    cui2node_id: Dict[str, int] = {cui: node_id for node_id, cui in enumerate(unique_cuis_set)}

    return cui2node_id


def create_relations2id_dicts(mrrel_df: pd.DataFrame):
    mrrel_df.REL.fillna("NAN", inplace=True)
    mrrel_df.RELA.fillna("NAN", inplace=True)
    rel2id = {rel: rel_id for rel_id, rel in enumerate(mrrel_df.REL.unique())}
    rela2id = {rela: rela_id for rela_id, rela in enumerate(mrrel_df.RELA.unique())}
    rel2id["LOOP"] = max(rel2id.values()) + 1
    rela2id["LOOP"] = max(rela2id.values()) + 1
    logging.info(f"There are {len(rel2id.keys())} unique RELs and {len(rela2id.keys())} unique RELAs")
    print("REL2REL_ID", rel2id)
    print("RELA2RELA_ID", rela2id)
    return rel2id, rela2id


def main(args):
    output_dir = args.output_dir
    if not os.path.exists(output_dir) and output_dir != '':
        os.makedirs(output_dir)

    unique_cuis_set, max_mention_length = get_text_dataset_unique_cuis_max_mention_length(
        input_dir=args.input_text_dataset_dir)

    logging.info("Loading MRCONSO....")
    mrconso_df = read_mrconso(args.mrconso)

    logging.info(f"Filtering non-English concept names. {mrconso_df.shape[0]} rows before filtration")
    mrconso_df = mrconso_df[mrconso_df["LAT"] == "ENG"]
    logging.info(f"Finished filtering non-English concept names. {mrconso_df.shape[0]} rows before filtration")
    logging.info(f"Removing duplicated (CUI, STR) pairs, {mrconso_df.shape[0]} rows before deletion")
    mrconso_df.drop_duplicates(subset=("CUI", "STR"), keep="first", inplace=True)
    logging.info(f"Removed duplicated (CUI, STR) pairs, {mrconso_df.shape[0]} rows after deletion")
    mrconso_df["STR"].fillna('', inplace=True)

    # split_val = args.split_val
    logging.info("Loading MRREL....")
    mrrel_df = read_mrrel(args.mrrel)[["CUI1", "CUI2", "REL", "RELA"]]
    logging.info(f"Filtering MRREL by CUI. Size before filtering: {mrrel_df.shape}")
    mrrel_df = mrrel_df[mrrel_df["CUI1"].isin(unique_cuis_set) | mrrel_df["CUI2"].isin(unique_cuis_set)]
    logging.info(f"Filtering MRREL by CUI. After filtering: {mrrel_df.shape}")

    if args.filter_cui_by_graph:
        mrrel_cui_set_1, mrrel_cui_set_2 = set(mrrel_df["CUI1"].unique()), set(mrrel_df["CUI2"].unique())
        mrrel_cui_set_1 = mrrel_cui_set_1.union(mrrel_cui_set_2)
        logging.info(f"Filtering MRCONSO by CUI. Size before filtering: {mrconso_df.shape}")
        mrconso_df = mrconso_df[mrconso_df["CUI"].isin(mrrel_cui_set_1)]
        logging.info(f"Size after CUI filtering: {mrconso_df.shape}")
        del mrrel_cui_set_1, mrrel_cui_set_2

    all_unique_cuis = set(mrconso_df["CUI"].unique())
    logging.info(f"# Unique CUIs in MRCONSO: {len(all_unique_cuis)}")
    cui2node_id = {cui: idx for idx, cui in enumerate(all_unique_cuis)}

    rel2id, rela2id = create_relations2id_dicts(mrrel_df)
    logging.info(f"There are {len(rel2id.keys())} RELs and {rela2id.keys()} RELAs")

    logging.info("Creating graph files")
    output_node_id2terms_list_path = os.path.join(output_dir, "node_id2terms_list")
    output_node_id2cui_path = os.path.join(output_dir, "id2cui")
    output_cui2node_id_path = os.path.join(output_dir, "cui2id")
    output_adjacency_lists_path = os.path.join(output_dir, "adjacency_lists")
    output_rel2rel_id_path = os.path.join(output_dir, f"rel2id")
    output_rela2rela_id_path = os.path.join(output_dir, f"rela2id")

    create_graph_files(mrconso_df=mrconso_df, mrrel_df=mrrel_df, rel2id=rel2id, rela2id=rela2id,
                       cui2node_id=cui2node_id,
                       output_node_id2terms_list_path=output_node_id2terms_list_path,
                       output_node_id2cui_path=output_node_id2cui_path,
                       output_adjacency_lists_path=output_adjacency_lists_path,
                       output_rel2rel_id_path=output_rel2rel_id_path,
                       output_cui2node_id_path=output_cui2node_id_path,
                       output_rela2rela_id_path=output_rela2rela_id_path, ignore_not_mapped_edges=True, )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--input_text_dataset_dir',
                        default="/home/c204/University/NLP/BERN2_sample/bert_reformated_to_umls/")
    parser.add_argument('--mrconso',
                        default="/home/c204/University/NLP/UMLS/2020AB/ENG_MRCONSO_FILTERED_BY_DATA.RRF")
    parser.add_argument('--mrrel',
                        default="/home/c204/University/NLP/UMLS/2020AB/MRREL.RRF")
    parser.add_argument('--filter_cui_by_graph', action="store_true")
    parser.add_argument('--output_dir', type=str,
                        default="/home/c204/University/NLP/BERN2_sample/debug_graph")
    args = parser.parse_args()
    main(args)
