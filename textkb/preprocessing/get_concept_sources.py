import logging
import os
import re
from argparse import ArgumentParser
from typing import Set, Tuple

from textkb.utils.io import create_dir_if_not_exists

CONCEPT_SOURCE_PATTERN = r'([A-Za-z0-9:"-_]+[:_][A-Za-z0-9:"-_]+)|CUI-less'
# ALLOWED_CONCEPT_ID_PATTERN = r'([A-Za-z0-9:"-_]+)(([,|][A-Za-z0-9:"]+)+)?'

def get_normalization_sources(anno_dir: str, field_sep: str, subfield_sep: str) -> Tuple[Set[str], Set[str]]:
    seen_sources: Set[str] = set()
    non_template_concept_ids: Set[str] = set()
    for annotation_filename in os.listdir(anno_dir):
        logging.info(f"Processing {annotation_filename}")
        anno_fpath = os.path.join(anno_dir, annotation_filename)
        with open(anno_fpath, 'r', encoding="utf-8") as inp_file:
            for line in inp_file:
                attrs = line.strip().split(field_sep)
                # print(line.strip())
                # print(attrs)
                # print('--')
                norm_concept_ids = attrs[1]
                for norm_concept_id in norm_concept_ids.split(subfield_sep):
                    if re.fullmatch(CONCEPT_SOURCE_PATTERN, norm_concept_id) is not None:
                        source = re.split(r"[:_]", norm_concept_id)[0]
                        seen_sources.add(source)
                    else:
                        if norm_concept_id.isdigit():
                            continue
                        non_template_concept_ids.add(norm_concept_id)
    return seen_sources, non_template_concept_ids


def main(args):
    seen_sources, non_template_concept_ids = get_normalization_sources(anno_dir=args.annotation_dir, field_sep="\t",
                                                                       subfield_sep='|')
    output_dir = args.output_dir
    create_dir_if_not_exists(output_dir)

    output_seen_sources_path = os.path.join(output_dir, "seen_sources.txt")
    output_non_template_concept_ids_path = os.path.join(output_dir, "non_template_normalization.txt")
    with open(output_seen_sources_path, 'w+', encoding="utf-8") as out_file:
        for source in seen_sources:
            out_file.write(f"{source}\n")
    with open(output_non_template_concept_ids_path, 'w+', encoding="utf-8") as out_file:
        for c_id in non_template_concept_ids:
            out_file.write(f"{c_id}\n")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--annotation_dir',
                        default="/home/c204/University/NLP/BERN2_sample/bert_reformated/annotations")
    parser.add_argument('--output_dir', default="delete/")
    args = parser.parse_args()
    main(args)
