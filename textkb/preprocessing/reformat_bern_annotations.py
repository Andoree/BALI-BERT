import json
import logging
import re
from argparse import ArgumentParser
from typing import Dict, List
import os

import spacy
from tqdm import tqdm

from textkb.utils.io import create_dir_if_not_exists

ALLOWED_CONCEPT_ID_PATTERN = r'([A-Za-z0-9:"-_]+)(([,|][A-Za-z0-9:"]+)+)?'


def sentenize_label_entities(tokenized_text, ann_list: List[Dict], subfield_sep):
    # doc = tokenizer(text, disable=("tagger", "entity", "ner", "lemmatizer", "trainable_lemmatizer", "pos_tagger",
    #                                "morphologizer", "entity_linker", "entity_ruler", "textcat", "tok2vec"))
    sent_id2text_and_span = {}
    for sent_id, sent in enumerate(tokenized_text.sents):
        sent_start, sent_end = sent.start_char, sent.end_char
        sent_text = sent.text
        sent_id2text_and_span[sent_id] = (sent_start, sent_end, sent_text)

    not_found_counter = 0
    for ann_dict in ann_list:
        fulldoc_entity_start, fulldoc_entity_end = (int(x) for x in ann_dict["span"].split(subfield_sep))
        entity_text = ann_dict["mention"]
        found_sentence = False

        for sent_id, (sent_start, sent_end, sent_text) in sent_id2text_and_span.items():
            if fulldoc_entity_start >= sent_start and fulldoc_entity_end <= sent_end:
                sentence_entity_start = fulldoc_entity_start - sent_start
                sentence_entity_end = fulldoc_entity_end - sent_start
                ann_dict["span"] = f"{sentence_entity_start}{subfield_sep}{sentence_entity_end}"

                assert entity_text in sent_text
                assert not found_sentence
                assert ann_dict.get("sentence_id") is None
                if ann_dict.get("sentence_id") is None:
                    ann_dict["sentence_id"] = sent_id
                found_sentence = True
        if not found_sentence:
            ann_dict["sentence_id"] = -1
            not_found_counter += 1
        if ann_dict.get("sentence_id") is None:
            ann_dict["sentence_id"] = -1
    return sent_id2text_and_span, not_found_counter


def reformat_annotation_dict(ann_dict: Dict, subfield_sep: str, field_sep: str) -> Dict[str, str]:
    # print(ann_dict)
    # sent_ids = ann_dict.get("sentence_ids", [])
    # ann_dict["sentence_ids"] = subfield_sep.join(str(x) for x in sent_ids)
    concept_id_list = ann_dict["id"]
    neural_flag = ann_dict["is_neural_normalized"]
    mention = ann_dict["mention"]
    assert field_sep not in mention
    mention_type = ann_dict["obj"]
    span_start = ann_dict["span"]["begin"]
    span_end = ann_dict["span"]["end"]
    new_concept_ids = []
    if len(concept_id_list) == 1 and isinstance(concept_id_list[0], list):
        concept_id_list = concept_id_list[0]
    for cons in concept_id_list:
        if re.fullmatch(ALLOWED_CONCEPT_ID_PATTERN, cons) is None:
            raise Exception(f"Unexpected concept id: {cons}")
        for con in re.split(r"[,|]", cons):
            new_concept_ids.append(con)
            assert subfield_sep not in con
    mention = mention.replace(field_sep, " ")

    assert field_sep not in mention
    assert field_sep not in mention_type

    ann_dict["id"] = subfield_sep.join(str(x) for x in new_concept_ids)
    ann_dict["span"] = f"{span_start}{subfield_sep}{span_end}"

    return ann_dict


def reformat_annotations(anno_list: List[Dict], subfield_sep: str, field_sep: str):
    return list(map(lambda x: reformat_annotation_dict(x, subfield_sep, field_sep), anno_list))


def process_document(json_doc: Dict, subfield_sep: str, field_sep: str) -> Dict:
    doc_text = json_doc["text"]
    doc_pubmed_id = json_doc["pmid"]
    anno_list = json_doc["annotations"]
    anno_list = reformat_annotations(anno_list, subfield_sep, field_sep)

    res = {
        "text": doc_text,
        "pubmed_id": doc_pubmed_id,
        "annotations": anno_list
    }
    return res


def ann_dict2str(ann_dict, sep):
    # print(ann_dict)
    assert len(ann_dict.keys()) in (6, 7, 8, 9)
    sentence_id = ann_dict["sentence_id"]
    concept_ids_str = ann_dict["id"]
    neural_flag = ann_dict["is_neural_normalized"]
    mention = ann_dict["mention"]
    assert sep not in mention
    mention_type = ann_dict["obj"]
    span = ann_dict["span"]
    mt = ann_dict.get("mutation_type", "")
    pm = ann_dict.get("ProteinMutation", "")
    nn = ann_dict.get("normalizedName", "")

    return f"{sentence_id}{sep}{concept_ids_str}{sep}{mention}{sep}{mention_type}{sep}{span}"


def reformat_bern_annotations(bern2_anno_dir: str, output_reformatted_dir: str, n_proc: int, subfield_sep: str = "|",
                              field_sep: str = "\t"):
    logging.info("Loading tokenizer")
    tokenizer = spacy.load("en_core_web_sm",
                           disable=("tagger", "entity", "ner", "lemmatizer", "trainable_lemmatizer", "pos_tagger",
                                    "morphologizer", "entity_linker", "entity_ruler", "textcat",))
    output_texts_dir = os.path.join(output_reformatted_dir, "texts/")
    output_annotations_dir = os.path.join(output_reformatted_dir, "annotations/")
    create_dir_if_not_exists(output_texts_dir)
    create_dir_if_not_exists(output_annotations_dir)

    seen_pubmed_ids = set()
    logging.info("Started processing annotation files")
    global_not_found_counter = 0
    entities_counter = 0
    # output_text_file_paths_list = []
    # output_ann_file_paths_list = []
    # pubmed_ids_list = []
    # sentence_texts_list = []
    # annotations_list = []
    for json_ann_filename in os.listdir(bern2_anno_dir):
        json_fname_attrs = json_ann_filename.split('.')
        logging.info(f'Processing {json_ann_filename}')
        assert len(json_fname_attrs) == 2 and json_fname_attrs[1] == "json"
        pubmed_file_id = json_fname_attrs[0]

        input_json_path = os.path.join(bern2_anno_dir, json_ann_filename)

        output_texts_file_path = os.path.join(output_texts_dir, f"{pubmed_file_id}.txt")
        output_ann_file_path = os.path.join(output_annotations_dir, f"{pubmed_file_id}.txt")
        pubmed_ids_list = []
        texts_list = []
        annotations_list = []
        with open(input_json_path, 'r', encoding="utf-8") as inp_file:
            for line in inp_file:
                json_doc = json.loads(line.strip())
                json_doc = process_document(json_doc, subfield_sep=subfield_sep, field_sep=field_sep)
                doc_text = json_doc["text"]
                ann_list = json_doc["annotations"]
                entities_counter += len(ann_list)
                doc_pubmed_id = json_doc["pubmed_id"]

                pubmed_ids_list.append(doc_pubmed_id)
                texts_list.append(doc_text)
                annotations_list.append(ann_list)
            tokenized_texts = tokenizer.pipe(texts_list, disable=(
                "tagger", "entity", "ner", "lemmatizer", "trainable_lemmatizer", "pos_tagger",
                "morphologizer", "entity_linker", "entity_ruler", "textcat",),
                                             n_process=n_proc)
            with open(output_texts_file_path, 'w+', encoding="utf-8") as out_texts_file, \
                    open(output_ann_file_path, 'w+', encoding="utf-8") as out_ann_file:
                for tokenized_text, doc_pubmed_id, ann_list in zip(tokenized_texts, pubmed_ids_list, annotations_list):
                    sent_id2text_and_span, not_found_counter = sentenize_label_entities(tokenized_text=tokenized_text,
                                                                                        ann_list=ann_list,
                                                                                        subfield_sep=subfield_sep)
                    global_not_found_counter += not_found_counter

                    seen_pubmed_ids.add(doc_pubmed_id)
                    for sent_id, (sent_start, sent_end, sent_text) in sent_id2text_and_span.items():
                        sent_text = sent_text.replace(field_sep, " ")
                        out_texts_file.write(f"{doc_pubmed_id}{subfield_sep}{sent_id}{field_sep}{sent_text}\n")

                    if len(ann_list) > 0:
                        ann_s = f'\n{doc_pubmed_id}{field_sep}'.join(
                            (ann_dict2str(ann_dict, field_sep) for ann_dict in ann_list))
                        ann_s = f"{doc_pubmed_id}{field_sep}{ann_s}\n"
                        out_ann_file.write(ann_s)
    logging.info(f"{global_not_found_counter} / {entities_counter} entities "
                 f"are not found in text after sentence tokenization")


def main(args):
    reformat_bern_annotations(bern2_anno_dir=args.bern2_anno_dir,
                              n_proc=args.n_proc,
                              output_reformatted_dir=args.output_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--bern2_anno_dir', default="/home/c204/University/NLP/BERN2_sample/bern_anno/")
    parser.add_argument('--n_proc', type=int, default=4)
    parser.add_argument('--output_dir', default="/home/c204/University/NLP/BERN2_sample/bert_reformated/")
    args = parser.parse_args()
    main(args)
