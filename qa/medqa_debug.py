import argparse
import json
import logging
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, BertConfig
from transformers import AutoTokenizer

from textkb.bert_modeling.multiple_choice_models import BertForMultipleChoiceWithMeanPooling
from textkb.utils.io import create_dir_if_not_exists


# from datasets import load_metric


class QuestionMultipleChoiceDataset(Dataset):

    def __init__(self, data_json, max_length, tokenizer):
        super(QuestionMultipleChoiceDataset).__init__()

        self.answer_choice_letters = ('A', 'B', 'C', 'D', 'E')
        self.choice2id = {ch: i for i, ch in enumerate(self.answer_choice_letters)}
        self.tokenizer = tokenizer
        self.sep_token = tokenizer.sep_token
        self.max_length = max_length
        self.tokenized_paired_samples = []
        labels = []

        for sample_dict in data_json:
            question = sample_dict["question"]
            options = tuple(map(lambda x: sample_dict["options"][x],
                                self.answer_choice_letters))

            samples = [[question, op] for op in options]
            for s in samples:
                print(">>", s)
            tokenized_sample = tokenizer.batch_encode_plus(samples,
                                                           max_length=self.max_length,
                                                           padding="max_length",
                                                           truncation=True,
                                                           return_tensors="pt")
            assert tokenized_sample["input_ids"].size() == (len(options), self.max_length)
            self.tokenized_paired_samples.append(tokenized_sample)
            labels.append(self.choice2id[sample_dict["answer_idx"]])
            print("<<", self.choice2id[sample_dict["answer_idx"]])
            print("<<<", tokenized_sample["input_ids"].size())
            print('--' * 10)
            ttt = [tokenizer.convert_ids_to_tokens(t, skip_special_tokens=False) for t in tokenized_sample["input_ids"]]
            for t in ttt:
                print('}', t)
            # ttt = [tokenizer.decode(t) for t in tokenized_sample["input_ids"]]
            # print(ttt)
            print('///' * 15)

        self.labels = torch.tensor(labels, dtype=torch.int64)

        assert len(self.tokenized_paired_samples) == len(self.labels)

    def __len__(self):
        return len(self.tokenized_paired_samples)

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokenized_paired_samples[idx]["input_ids"],
            "attention_mask": self.tokenized_paired_samples[idx]["attention_mask"],
            "labels": self.labels[idx]}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return {"accuracy": float(
        accuracy_score(labels, predictions, normalize=True, sample_weight=None))}


def load_jsonl_data(inp_path: str):
    with open(inp_path, 'r') as json_file:
        json_list = list(map(json.loads, json_file))
    assert isinstance(json_list[0], dict)

    return json_list


def main(args):
    input_data_dir = args.input_data_dir
    base_model_name = args.base_model_name
    max_length = args.max_length

    train_json_path = os.path.join(input_data_dir, "train.jsonl")
    dev_json_path = os.path.join(input_data_dir, "dev.jsonl")
    test_json_path = os.path.join(input_data_dir, "test.jsonl")

    train_data = load_jsonl_data(train_json_path)
    dev_data = load_jsonl_data(dev_json_path)
    test_data = load_jsonl_data(test_json_path)
    logging.info(f"Loaded data. Train - {len(train_data)}, Dev - {len(dev_data)}, Test - {len(test_data)}, "
                 f"Total - {len(train_data) + len(dev_data) + len(test_data)}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    train_dataset = QuestionMultipleChoiceDataset(train_data, max_length=max_length,
                                                  tokenizer=tokenizer)
    # dev_dataset = QuestionMultipleChoiceDataset(dev_data, max_length=max_length,
    #                                             tokenizer=tokenizer)
    # test_dataset = QuestionMultipleChoiceDataset(test_data, max_length=max_length,
    #                                              tokenizer=tokenizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', type=str, required=False,
                        default="/home/c204/University/NLP/text_kb/MedQA_US/")
    parser.add_argument('--max_length', type=int, required=False, default=256)
    parser.add_argument('--base_model_name', type=str, required=False,
                        default="michiyasunaga/BioLinkBERT-base")
    # parser.add_argument('--batch_size', type=int, required=False, default=16)
    # parser.add_argument('--gradient_accumulation_steps', type=int, required=False, default=1)
    # parser.add_argument('--max_length', type=int, required=False, default=512)
    # parser.add_argument('--bert_pooling', type=str, required=False, choices=("cls", "mean"), default="cls")
    # parser.add_argument('--model_config_path', type=str, required=False)
    # parser.add_argument('--num_epochs', type=int, required=False, default=50)
    # parser.add_argument('--warmup_ratio', type=float, required=False, default=0.1)
    # parser.add_argument('--warmup_steps', type=int, required=False, default=0)
    # parser.add_argument('--learning_rate', type=float, required=False, default=1e-5)
    # parser.add_argument("--freeze_embs", action="store_true")
    # parser.add_argument("--freeze_layers", required=False, type=int)
    # parser.add_argument("--fp16", action="store_true")
    # parser.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()
    main(args)
