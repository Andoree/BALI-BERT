import logging
from typing import List, Set, Iterable
import os

from tqdm import tqdm


def create_sentences_word_vocab_spacy(input_sentences_dir, word_tokenizer, n_proc: int,
                                      tok_pipeline_disable: Iterable[str],
                                      field_sep='\t'):
    dataset_vocab: Set[str] = set()
    n_files = len(list(os.listdir(input_sentences_dir)))
    for fname in tqdm(os.listdir(input_sentences_dir), total=n_files):
        input_sentences_path = os.path.join(input_sentences_dir, fname)
        sentences_list = []
        # print(fname)
        with open(input_sentences_path, 'r', encoding="utf-8") as inp_file:
            for line in inp_file:
                # print(line.strip())
                attrs = line.strip().split(field_sep)
                assert len(attrs) <= 2
                if len(attrs) == 1:
                    continue
                sentence_text = attrs[1]
                sentences_list.append(sentence_text)
            tokenized_texts = word_tokenizer.pipe(sentences_list, disable=tok_pipeline_disable,
                                                  n_process=n_proc)
            file_vocab = create_vocab_from_spacy_docs(tokenized_texts)
            dataset_vocab.update(file_vocab)
    return dataset_vocab


def create_sentences_word_vocab_nltk(input_sentences_dir, word_tokenizer, do_lower_case, field_sep='\t'):
    dataset_vocab: Set[str] = set()
    n_files = len(list(os.listdir(input_sentences_dir)))
    for fname in tqdm(os.listdir(input_sentences_dir), total=n_files):
        input_sentences_path = os.path.join(input_sentences_dir, fname)
        with open(input_sentences_path, 'r', encoding="utf-8") as inp_file:
            for line in inp_file:
                attrs = line.strip().split(field_sep)
                assert len(attrs) <= 2
                if len(attrs) == 1:
                    continue
                sentence_text = attrs[1]
                if do_lower_case:
                    words = (sentence_text[sp_s:sp_e].lower() for (sp_s, sp_e) in
                             word_tokenizer.span_tokenize(sentence_text))
                else:
                    words = (sentence_text[sp_s:sp_e] for (sp_s, sp_e) in word_tokenizer.span_tokenize(sentence_text))
                dataset_vocab.update(words)

    return dataset_vocab


def create_vocab_from_spacy_docs(docs_list) -> Set[str]:
    vocab = set()
    for doc in docs_list:
        tokens = set((t.text for t in doc))
        vocab.update(tokens)
    logging.info(f"Created word vocab. The size is {len(vocab)}")
    return vocab


def transformer_tokenize_str_list(string_list: List[str], transformer_tokenizer, add_special_tokens=False,
                                  max_length=None, truncation=False, return_attention_mask=False,
                                  return_token_type_ids=False):
    input_ids = transformer_tokenizer(string_list,
                                      add_special_tokens=add_special_tokens,
                                      return_attention_mask=return_attention_mask,
                                      max_length=max_length,
                                      truncation=truncation,
                                      return_token_type_ids=return_token_type_ids)["input_ids"]
    return input_ids


def transformer_vocab2input_ids(vocab: Set[str], transformer_tokenizer, add_special_tokens=False,
                                return_attention_mask=False,
                                return_token_type_ids=False):
    id2word = list(vocab)
    input_ids = transformer_tokenize_str_list(id2word, transformer_tokenizer,
                                              add_special_tokens=add_special_tokens,
                                              return_attention_mask=return_attention_mask,
                                              return_token_type_ids=return_token_type_ids, )
    assert len(input_ids) == len(id2word)
    vocab2input_ids = {word: inp_ids for word, inp_ids in zip(id2word, input_ids)}

    return vocab2input_ids
