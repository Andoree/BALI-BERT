import logging
import os
from argparse import ArgumentParser
from typing import List, Tuple, Set, Iterable

from tqdm import tqdm
from transformers import AutoTokenizer
# from nltk import word_tokenize
from nltk.tokenize import TreebankWordTokenizer
from textkb.data.entity import Entity
from textkb.utils.io import create_dir_if_not_exists, load_entities_groupby_doc_id_sent_id, load_dict, save_dict
from textkb.utils.tokenization import create_sentences_word_vocab_nltk, transformer_vocab2input_ids


def get_word_concept_labels_list_spacy(sent_entities: List[Entity], word_token):
    # DEPRECATED
    word_text = word_token.text
    word_span_start, word_span_end = word_token.idx, word_token.idx + len(word_text)
    word_labels = []
    for entity in sent_entities:
        e_s_start, e_s_end = entity.span_start, entity.span_end
        if word_span_start >= e_s_start and word_span_end <= e_s_end:
            assert word_text in entity.mention_str
            labels = (n_id for n_id in entity.node_ids if n_id not in ("CUILESS", "CUI-LESS"))
            word_labels.extend(labels)
            break
        else:
            assert word_text not in entity.mention_str

    return word_labels


def get_word_concept_labels_list(sent_entities: List[Entity], word, span_start, span_end):
    word_labels = []
    for entity in sent_entities:
        e_s_start, e_s_end = entity.span_start, entity.span_end
        if span_start >= e_s_start and span_end <= e_s_end:
            assert word in entity.mention_str
            labels = (n_id for n_id in entity.node_ids if n_id not in ("CUILESS", "CUI-LESS"))
            word_labels.extend(labels)
            # break
        else:
            pass

    return word_labels


def truncate_incomplete_entities(input_ids, token_level_concept_labels, truncated_entity_ids: Set[int]):
    num_input_ids = len(input_ids)
    num_trunc_ent_labels = len(truncated_entity_ids)
    pointer = num_input_ids
    for i in range(num_input_ids - 1, -1, -1):
        token_labels = token_level_concept_labels[i]
        if len(token_labels) == 1 and token_labels[0] == -1:
            break
        if len(truncated_entity_ids.intersection(set(token_labels))) == num_trunc_ent_labels:
            pointer = i
        else:
            break
    return pointer


def word_tokenize_label_subword_tokens(word_spans, sentence_text, vocab2input_ids, sentence_max_length: int,
                                       sentence_entities: List[Entity], mask_entities, drop_sent_without_entities,
                                       do_lower_case, mask_token_id: int, cls_token_id: int, sep_token_id):
    if drop_sent_without_entities and len(sentence_entities) == 0:
        return None

    input_ids = [cls_token_id, ]
    token_ent_binary_mask = [0, ]
    edge_index_token_idx = []
    edge_index_entity_idx = []
    num_entity_subtokens = 0
    token_level_concept_labels = [(-1,), ]
    truncated_entity_ids = None

    for (sp_s, sp_e) in word_spans:
        word = sentence_text[sp_s:sp_e]
        if do_lower_case:
            word = word.lower()

        word_node_ids = get_word_concept_labels_list(sent_entities=sentence_entities,
                                                     word=word,
                                                     span_start=sp_s,
                                                     span_end=sp_e)
        # Tokenize the word and count # of subwords
        word_inp_ids = vocab2input_ids[word]
        n_subwords = len(word_inp_ids)
        # print(token_level_concept_labels)

        if len(input_ids) + n_subwords > sentence_max_length - 1:
            if len(word_node_ids) == 0:
                # Truncated word is not an entity
                break
            else:
                # Truncated word is a part of an entity
                truncated_entity_ids = set(word_node_ids)
                break
        if len(word_node_ids) == 0:
            m = 0
            token_level_concept_labels.extend(((-1,),) * n_subwords, )
        else:
            if mask_entities:
                word_inp_ids = [mask_token_id, ] * n_subwords
            m = 1
            num_entity_subtokens += n_subwords
            token_idx_start_v1 = max(edge_index_token_idx) + 1 if len(edge_index_token_idx) > 0 else 0
            token_idx_start_v2 = sum(token_ent_binary_mask)
            assert token_idx_start_v1 == token_idx_start_v2
            token_idx_start = token_idx_start_v1
            token_idx_end = token_idx_start + n_subwords
            for w_n_id in word_node_ids:
                edge_index_token_idx.extend(range(token_idx_start, token_idx_end))
                edge_index_entity_idx.extend([w_n_id, ] * n_subwords)
                # sentence_unique_entity_node_ids.add(w_n_id)

            token_level_concept_labels.extend((word_node_ids,) * n_subwords)
        input_ids.extend(word_inp_ids)
        token_ent_binary_mask.extend([m, ] * n_subwords)

    assert len(input_ids) == len(token_level_concept_labels) == len(token_ent_binary_mask)
    assert len(edge_index_token_idx) == len(edge_index_entity_idx)

    max_edge_index_token_idx = max(edge_index_token_idx) if len(edge_index_token_idx) > 0 else -1
    assert max_edge_index_token_idx == sum(token_ent_binary_mask) - 1

    if truncated_entity_ids is not None:
        sent_num_entity_tokens = sum(token_ent_binary_mask)
        if sent_num_entity_tokens == 0:
            assert len(edge_index_token_idx) == len(edge_index_entity_idx) == 0
        num_inp_ids_before_truncation = len(input_ids)
        truncation_pointer = truncate_incomplete_entities(input_ids, token_level_concept_labels,
                                                          truncated_entity_ids=truncated_entity_ids)
        num_inp_ids_after_truncation = len(input_ids)
        inp_ids_delta = num_inp_ids_before_truncation - num_inp_ids_after_truncation
        assert inp_ids_delta >= 0
        if inp_ids_delta > 0:
            print("inp_ids_delta", inp_ids_delta)

        num_edges_before_trunc = len(edge_index_token_idx)
        num_edges_to_delete = inp_ids_delta * len(truncated_entity_ids)
        num_edges_after_trunc = num_edges_before_trunc - num_edges_to_delete
        assert num_edges_after_trunc >= 0
        # edge_index_token_idx = edge_index_token_idx[:num_edges_after_trunc]
        # edge_index_entity_idx = edge_index_entity_idx[:num_edges_after_trunc]
        if truncation_pointer == 1:
            return None
        input_ids = input_ids[:truncation_pointer]
        token_ent_binary_mask = token_ent_binary_mask[:truncation_pointer]

        sent_num_entity_tokens = sum(token_ent_binary_mask)
        edge_index_token_idx = [x for x in edge_index_token_idx if x < sent_num_entity_tokens]
        edge_index_entity_idx = edge_index_entity_idx[:len(edge_index_token_idx)]

    sent_num_entity_tokens = sum(token_ent_binary_mask)
    if sent_num_entity_tokens == 0:
        return None
    input_ids.append(sep_token_id)
    token_ent_binary_mask.append(0)
    n_sent_tokens = len(input_ids)
    assert n_sent_tokens <= sentence_max_length

    d = {"input_ids": input_ids,
         "token_ent_binary_mask": token_ent_binary_mask,
         "edge_index_token_idx": edge_index_token_idx,
         "edge_index_entity_idx": edge_index_entity_idx,
         # "local_n_id2global_n_id": local_n_id2global_n_id
         }
    return d


def tokenize_sentences(input_sentences_dir: str, input_anno_dir: str, word_tokenizer, transformer_tokenizer,
                       output_dir: str, cui2node_id, mask_entities: bool, drop_sent_without_entities: bool,
                       sentence_max_length: int, drop_cuiless: bool, do_lower_case: bool,
                       field_sep="\t", subfield_sep="|"):
    mask_token_id: int = transformer_tokenizer.mask_token_id
    cls_token_id: int = transformer_tokenizer.cls_token_id
    sep_token_id: int = transformer_tokenizer.sep_token_id

    logging.info("Creating word dataset's word-level vocabulary ")
    dataset_vocab = create_sentences_word_vocab_nltk(input_sentences_dir=input_sentences_dir,
                                                     word_tokenizer=word_tokenizer,
                                                     do_lower_case=do_lower_case
                                                     )
    logging.info(f"Dataset's word vocab size: {len(dataset_vocab)}")
    logging.info("Tokenizing word vocabulary with Transformers tokenizer....")
    vocab2input_ids = transformer_vocab2input_ids(vocab=dataset_vocab, transformer_tokenizer=transformer_tokenizer, )
    logging.info("Finished tokenizing word vocabulary.")
    del dataset_vocab

    n_files = len(list(os.listdir(input_sentences_dir)))
    logging.info(f"Processing dataset...")
    for fname in tqdm(os.listdir(input_sentences_dir), total=n_files):
        input_sentences_path = os.path.join(input_sentences_dir, fname)
        input_annotations_path = os.path.join(input_anno_dir, fname)
        doc_sent_id2entities = load_entities_groupby_doc_id_sent_id(input_path=input_annotations_path,
                                                                    field_sep=field_sep,
                                                                    cui2node_id=cui2node_id,
                                                                    drop_cuiless=drop_cuiless,
                                                                    do_lower_case=do_lower_case,
                                                                    subfield_sep=subfield_sep)

        output_preprocessed_output_path = os.path.join(output_dir, fname)
        with open(input_sentences_path, 'r', encoding="utf-8") as inp_file, \
                open(output_preprocessed_output_path, 'w+', encoding="utf-8") as out_file:

            for line in inp_file:
                attrs = line.strip().split(field_sep)
                assert len(attrs) <= 2
                if len(attrs) == 1:
                    continue
                doc_id_sent_id = attrs[0]
                sentence_text = attrs[1]
                doc_id, sentence_id = (int(x) for x in doc_id_sent_id.split(subfield_sep))
                if doc_sent_id2entities.get(doc_id) is None:
                    continue
                lst = doc_sent_id2entities[doc_id].get(sentence_id, [])
                if len(lst) == 0:
                    continue
                # sentences_list.append(sentence_text)
                # doc_ids_list.append(doc_id)
                # sent_ids_list.append(sentence_id)
                word_spans: List[Tuple[int, int]] = word_tokenizer.span_tokenize(sentence_text)

                sentence_entities = doc_sent_id2entities[doc_id][sentence_id]
                d = word_tokenize_label_subword_tokens(word_spans=word_spans,
                                                       sentence_text=sentence_text,
                                                       vocab2input_ids=vocab2input_ids,
                                                       sentence_max_length=sentence_max_length,
                                                       sentence_entities=sentence_entities,
                                                       mask_entities=mask_entities,
                                                       do_lower_case=do_lower_case,
                                                       drop_sent_without_entities=drop_sent_without_entities,
                                                       cls_token_id=cls_token_id, mask_token_id=mask_token_id,
                                                       sep_token_id=sep_token_id)
                if d is None:
                    continue
                input_ids = d["input_ids"]
                token_ent_binary_mask = d["token_ent_binary_mask"]
                edge_index_token_idx = d["edge_index_token_idx"]
                edge_index_entity_idx = d["edge_index_entity_idx"]

                ii_s = ','.join(str(x) for x in input_ids)
                tebm_s = ','.join(str(x) for x in token_ent_binary_mask)
                eiti_s = ','.join(str(x) for x in edge_index_token_idx)
                eiei_s = ','.join(str(x) for x in edge_index_entity_idx)
                s = (f"{doc_id}{subfield_sep}{sentence_id}{field_sep}"
                     f"{ii_s}{field_sep}{tebm_s}{field_sep}{eiti_s}{field_sep}{eiei_s}\n")
                out_file.write(s)


def main(args):
    output_dir = args.output_dir
    create_dir_if_not_exists(output_dir)
    args_fname = os.path.join(output_dir, "config.txt")

    save_dict(save_path=args_fname, dictionary=vars(args), )

    cui2node_id_path = args.cui2node_id_path
    cui2node_id = load_dict(cui2node_id_path, dtype_1=str, dtype_2=int)

    word_tokenizer = TreebankWordTokenizer()
    transformer_tokenizer = AutoTokenizer.from_pretrained(args.transformer_tokenizer_name,
                                                          do_lower_case=args.do_lower_case)
    tokenize_sentences(input_sentences_dir=args.input_sentences_dir,
                       input_anno_dir=args.input_anno_dir,
                       word_tokenizer=word_tokenizer,
                       transformer_tokenizer=transformer_tokenizer,
                       output_dir=output_dir,
                       cui2node_id=cui2node_id,
                       mask_entities=args.mask_entities,
                       drop_sent_without_entities=args.drop_sent_without_entities,
                       sentence_max_length=args.sentence_max_length,
                       do_lower_case=args.do_lower_case,
                       drop_cuiless=args.drop_cuiless)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
    parser = ArgumentParser()
    parser.add_argument('--input_sentences_dir',)
    parser.add_argument('--input_anno_dir',)
    parser.add_argument('--cui2node_id_path',)
    parser.add_argument('--transformer_tokenizer_name', default="prajjwal1/bert-tiny")
    parser.add_argument('--output_dir',)
    parser.add_argument('--sentence_max_length', type=int,)
    parser.add_argument('--drop_sent_without_entities', action="store_true")
    parser.add_argument('--do_lower_case', action="store_true")
    parser.add_argument('--mask_entities', action="store_true")
    parser.add_argument('--drop_cuiless', action="store_true")

    # parser.add_argument('--input_sentences_dir',
    #                     default="/home/c204/University/NLP/BERN2_sample/bert_reformated/texts")
    # parser.add_argument('--input_anno_dir',
    #                     default="/home/c204/University/NLP/BERN2_sample/bert_reformated_to_umls/")
    # parser.add_argument('--cui2node_id_path',
    #                     default="/home/c204/University/NLP/BERN2_sample/debug_graph/cui2id")
    # parser.add_argument('--transformer_tokenizer_name', default="prajjwal1/bert-tiny")
    # parser.add_argument('--output_dir', default="/home/c204/University/NLP/BERN2_sample/TOKENIZED_UNMASKED/")
    # parser.add_argument('--sentence_max_length', type=int,
    #                     default=48)
    # # parser.add_argument('--n_proc', type=int,
    # #                     default=4)
    # parser.add_argument('--drop_sent_without_entities', default=True)
    # parser.add_argument('--do_lower_case', default=True)
    # parser.add_argument('--mask_entities', default=False)
    # parser.add_argument('--drop_cuiless', default=True)
    args = parser.parse_args()
    main(args)
