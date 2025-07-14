import itertools
import json
import logging
import os
import re
from argparse import ArgumentParser
from typing import Dict

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from textkb.utils.io import read_mrconso, create_dir_if_not_exists, read_mrrel

PRESENT_ANNOTATION_SOURCES = ("cellosaurus", "NCBIGene", "CL", "mim", "CUI-less", "mesh", "OMIM",
                              "CHEBI", "NCBITaxon", "EntrezGene", "omim", "NCBI", "MESH", "CVCL")

SOURCE_ID2UMLS_SOURCE = {
    "cellosaurus": "NCI_CELLOSAURUS",
    # "NCBIGene": "NCBI",
    # "CL": "",
    "mim": "OMIM",
    "CUI-less": "CUI-less",
    "mesh": "MSH",
    # "CHEBI": "",
    "NCBITaxon": "NCBI",
    # "EntrezGene": "",
    "omim": "OMIM",
    "OMIM": "OMIM",
    "NCBI": "NCBI",
    "MESH": "MSH",
    "CVCL": "NCI_CELLOSAURUS"
    # CVCL
}

NORM_TEMPLATE = (r"(?P<sab>MESH|mesh|omim|mim|OMIM|cellosaurus|NCBITaxon|NCBI|NCBIGene|CHEBI|CL|EntrezGene|CVCL)[:_]"
                 r"(txid)?(?P<cid>[0-9a-zA-Z_]+)")


def load_cui_mention_from_json(json_path, source_id2cui):
    data = json.load(open(json_path))
    cui_mention_pairs = []
    for data_dict in data["annotations"]:
        # print("data_dict", data_dict)
        ids = data_dict["id"]
        mention = data_dict["mention"]
        for cid in ids:
            m = re.fullmatch(NORM_TEMPLATE, cid)
            if m is not None:
                current_source_name = m.group("sab")
                local_cid = m.group("cid")
                assert current_source_name in PRESENT_ANNOTATION_SOURCES
            else:
                pass

            sab = SOURCE_ID2UMLS_SOURCE[current_source_name]
            # umls_cui = source_id2cui[sab].get(local_cid, "CUILESS")
            umls_cui = source_id2cui[sab][local_cid]
            cui_mention_pairs.append((umls_cui, mention))
    return cui_mention_pairs


def create_source_concept_id2cui_map(mrconso_df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    source_id2cui: Dict[str, Dict[str, str]] = {}
    allowed_sab_set = set(SOURCE_ID2UMLS_SOURCE.values())
    for i, row in mrconso_df.iterrows():
        sab = row["SAB"]
        code = row["CODE"]
        cui = row["CUI"]
        assert sab in allowed_sab_set
        if source_id2cui.get(sab) is None:
            source_id2cui[sab] = {}
        source_id2cui[sab][str(code)] = cui
    return source_id2cui


def main(args):
    json_path = args.json_path
    mrconso_path = args.mrconso
    mrrel_path = args.mrrel
    output_dir = args.output_dir
    create_dir_if_not_exists(output_dir)
    logging.info(f"Loading MRCONSO...")
    mrconso_df = read_mrconso(mrconso_path, usecols=("CUI", "STR", "SAB", "CODE")).drop_duplicates()
    logging.info(f"Loaded MRCONSO. Shape: {mrconso_df.shape}")

    keep_sources_list = SOURCE_ID2UMLS_SOURCE.values()
    logging.info(f"Filtering MRCONSO")
    mrconso_df = mrconso_df[mrconso_df.SAB.isin(keep_sources_list)]
    logging.info(f"MRCONSO: {mrconso_df.shape}")

    logging.info(f"Loading MRREL...")
    mrrel_df = read_mrrel(mrrel_path, usecols=("CUI1", "CUI2", "REL", "RELA", "SAB")) \
        .drop_duplicates(subset=["CUI1", "CUI2", "REL", "RELA"])
    logging.info(f"Loaded MRREL. Shape: {mrrel_df.shape}")
    mrrel_df = mrrel_df[mrrel_df["REL"].isin(["RN", "RB", "PAR", "CHD"])]
    logging.info(f"Filtered MRREL by REL type (kept RN, RB, PAR, CHD only). Shape: {mrrel_df.shape}")
    print("MRREL examples:\n", mrrel_df.head(5))
    code2cui_dict = create_source_concept_id2cui_map(mrconso_df=mrconso_df)

    cui_mention_pairs = load_cui_mention_from_json(json_path, code2cui_dict)
    print("Pairs:")
    for p in cui_mention_pairs:
        print(p)
    cui2name = {cui: m for cui, m in cui_mention_pairs}

    selected_node_names = [f"{cui}|{name}" for cui, name in cui2name.items()]
    for _, row in mrconso_df.iterrows():
        cui = row["CUI"]
        name = row["STR"]
        if cui2name.get(cui) is None:
            cui2name[cui] = name
    node_names = [f"{cui}|{name}" for cui, name in cui2name.items()]
    # del cui2name
    # gc.collect()
    logging.info("Creating Networkx graph...")
    G = nx.Graph()
    G.add_nodes_from(node_names)
    logging.info("Reading edges from MRREL...")
    dropped_edges = 0
    for _, row in tqdm(mrrel_df.iterrows(), total=mrrel_df.shape[0], mininterval=10.0):
        cui_1 = row["CUI1"]
        cui_2 = row["CUI2"]
        if cui2name.get(cui_1) is None or cui2name.get(cui_2) is None:
            dropped_edges += 1
            continue

        cui_1 = f"{cui_1}|{cui2name[cui_1]}"
        cui_2 = f"{cui_2}|{cui2name[cui_2]}"
        G.add_edge(cui_1, cui_2)
    dropped_edges += 1
    logging.info(f"Dropped {dropped_edges} edges")
    logging.info("Calculating shortest paths")
    output_paths_file = os.path.join(output_dir, "paths.txt")
    with open(output_paths_file, 'w', encoding="utf-8") as out_file:
        for (node_name_1, node_name_2) in tqdm(itertools.combinations(selected_node_names, 2),
                                               total=(len(selected_node_names) * (len(selected_node_names) - 1)) // 2):
            try:
                path = nx.shortest_path(G, source=node_name_1, target=node_name_2)
                result_graph = G.subgraph(path)

                nx.draw(result_graph, node_size=20, node_color='blue', font_size=10, font_weight='bold',
                        with_labels=True)
                output_graph_path = os.path.join(output_dir, f"path_{node_name_1}-{node_name_2}.png")
                plt.savefig(output_graph_path, format="PNG")
                path_s = " -- ".join(result_graph)
                out_file.write(f"{path_s}\n")
            except nx.NetworkXNoPath as e:
                print(f"Failed finding path between nodes {node_name_1} and {node_name_2}.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--mrconso', default="/home/c204/University/NLP/UMLS/2020AB/ENG_MRCONSO.RRF")
    parser.add_argument('--mrrel', default="/home/c204/University/NLP/UMLS/2020AB/MRREL.RRF")
    parser.add_argument('--json_path',
                        default="/home/c204/University/NLP/text_kb/textkb/textkb/mesh_paths/mesh_samples.json")
    parser.add_argument('--output_dir', default="paths/")

    args = parser.parse_args()
    main(args)
