import logging
import os
import re
from argparse import ArgumentParser
from typing import Dict

import pandas as pd

from textkb.utils.io import read_mrconso, create_dir_if_not_exists

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


def remap_annotation2umls(in_anno_dir: str, out_anno_dir: str, out_anno_not_matched_dir: str,
                          source_id2cui: Dict[str, Dict[str, str]], field_sep: str, subfield_sep: str):
    create_dir_if_not_exists(out_anno_dir)
    create_dir_if_not_exists(out_anno_not_matched_dir)
    originally_cuiless_counter = 0
    mapping_failed_counter = 0
    mapping_successful_counter = 0
    unsupported_source_counter = 0
    for anno_fname in os.listdir(in_anno_dir):
        logging.info(f"Processing {anno_fname}")
        in_anno_fpath = os.path.join(in_anno_dir, anno_fname)
        out_anno_fpath = os.path.join(out_anno_dir, anno_fname)
        out_unmatched_anno_fpath = os.path.join(out_anno_not_matched_dir, anno_fname)
        with open(in_anno_fpath, 'r', encoding="utf-8") as in_file, \
                open(out_anno_fpath, 'w', encoding="utf-8") as out_file, \
                open(out_unmatched_anno_fpath, 'w', encoding="utf-8") as out_unmatched_file:
            for line in in_file:
                # print(line.strip())
                attrs = line.strip().split(field_sep)
                source_concept_ids = attrs[2].split(subfield_sep)
                current_source_name = None
                umls_cuis = set()
                for scid in source_concept_ids:
                    if scid == "CUI-less":
                        originally_cuiless_counter += 1
                        umls_cuis.add(scid)
                        continue
                    m = re.fullmatch(NORM_TEMPLATE, scid)
                    if m is not None:
                        current_source_name = m.group("sab")
                        local_cid = m.group("cid")
                        if "txid" in current_source_name:
                            current_source_name = "NCBITaxon"
                        assert current_source_name in PRESENT_ANNOTATION_SOURCES
                    else:
                        pass
                    if current_source_name is None:
                        assert m is not None
                        # Sanity check: no unexpected sources are allowed
                        assert current_source_name in PRESENT_ANNOTATION_SOURCES
                    sab = SOURCE_ID2UMLS_SOURCE.get(current_source_name, None)
                    if sab is None:
                        umls_cuis.add("CUILESS")
                        out_unmatched_file.write(line)
                        unsupported_source_counter += 1
                        continue
                    umls_cui = source_id2cui[sab].get(local_cid, "CUILESS")
                    if umls_cui == "CUILESS":
                        out_unmatched_file.write(line)
                        mapping_failed_counter += 1
                    else:
                        mapping_successful_counter += 1
                    umls_cuis.add(umls_cui)
                new_concept_ids = subfield_sep.join(umls_cuis)
                attrs[2] = new_concept_ids
                s = field_sep.join(attrs)
                out_file.write(f"{s}\n")
    logging.info(f"mapping_successful_counter: {mapping_successful_counter}\n"
                 f"mapping_failed_counter: {mapping_failed_counter}\n"
                 f"originally_cuiless_counter: {originally_cuiless_counter}")


def main(args):
    logging.info(f"Loading MRCONSO")
    mrconso_df = read_mrconso(args.mrconso, usecols=("CUI", "STR", "SAB", "CODE"))

    keep_sources_list = SOURCE_ID2UMLS_SOURCE.values()
    logging.info(f"Filtering MRCONSO")
    mrconso_df = mrconso_df[mrconso_df.SAB.isin(keep_sources_list)]

    code2cui_dict = create_source_concept_id2cui_map(mrconso_df=mrconso_df)
    remap_annotation2umls(in_anno_dir=args.annotation_dir, out_anno_dir=args.output_dir,
                          out_anno_not_matched_dir=args.output_unmatched_dir, source_id2cui=code2cui_dict,
                          field_sep="\t", subfield_sep='|')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--mrconso', default="/home/c204/University/NLP/UMLS/2020AB/ENG_MRCONSO_FILTERED.RRF")
    parser.add_argument('--annotation_dir',
                        default="/home/c204/University/NLP/BERN2_sample/bert_reformated/annotations/")
    parser.add_argument('--output_dir', default="/home/c204/University/NLP/BERN2_sample/bert_reformated_to_umls/")
    parser.add_argument('--output_unmatched_dir',
                        default="/home/c204/University/NLP/BERN2_sample/bert_reformated_unmatched_umls/")

    args = parser.parse_args()
    main(args)
