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

    def __init__(self, data_json, max_length, tokenizer, answer_choice_letters):
        super(QuestionMultipleChoiceDataset).__init__()

        # self.answer_choice_letters = ('A', 'B', 'C', 'D', 'E')
        self.answer_choice_letters = answer_choice_letters
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
            tokenized_sample = tokenizer.batch_encode_plus(samples,
                                                           max_length=self.max_length,
                                                           padding="max_length",
                                                           truncation=True,
                                                           return_tensors="pt")
            assert tokenized_sample["input_ids"].size() == (len(options), self.max_length)
            self.tokenized_paired_samples.append(tokenized_sample)
            labels.append(self.choice2id[sample_dict["answer_idx"]])

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
    max_length = args.max_length
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    num_epochs = args.num_epochs
    use_5_options = args.use_5_options
    learning_rate = args.learning_rate
    bert_pooling = args.bert_pooling
    model_config_path = args.model_config_path
    warmup_ratio = args.warmup_ratio
    warmup_steps = args.warmup_steps
    base_model_name = args.base_model_name
    freeze_embs = args.freeze_embs
    freeze_layers = args.freeze_layers
    fp16 = args.fp16
    output_dir = args.output_dir
    output_finetuned_dir = os.path.join(output_dir, "finetuned_models/")
    create_dir_if_not_exists(output_finetuned_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if not use_5_options:
        answer_choice_letters = ('A', 'B', 'C', 'D')

        train_json_path = os.path.join(input_data_dir, "4_options/", "phrases_no_exclude_train.jsonl")
        dev_json_path = os.path.join(input_data_dir, "4_options/", "phrases_no_exclude_dev.jsonl")
        test_json_path = os.path.join(input_data_dir, "4_options/", "phrases_no_exclude_test.jsonl")

    else:
        answer_choice_letters = ('A', 'B', 'C', 'D', 'E')

        train_json_path = os.path.join(input_data_dir, "train.jsonl")
        dev_json_path = os.path.join(input_data_dir, "dev.jsonl")
        test_json_path = os.path.join(input_data_dir, "test.jsonl")

    train_data = load_jsonl_data(train_json_path)
    dev_data = load_jsonl_data(dev_json_path)
    test_data = load_jsonl_data(test_json_path)
    logging.info(f"Loaded data. Train - {len(train_data)}, Dev - {len(dev_data)}, Test - {len(test_data)}, "
                 f"Total - {len(train_data) + len(dev_data) + len(test_data)}")

    train_dataset = QuestionMultipleChoiceDataset(train_data, max_length=max_length,
                                                  tokenizer=tokenizer,
                                                  answer_choice_letters=answer_choice_letters)
    dev_dataset = QuestionMultipleChoiceDataset(dev_data, max_length=max_length,
                                                tokenizer=tokenizer,
                                                answer_choice_letters=answer_choice_letters)
    test_dataset = QuestionMultipleChoiceDataset(test_data, max_length=max_length,
                                                 tokenizer=tokenizer,
                                                 answer_choice_letters=answer_choice_letters)

    if bert_pooling == "cls":
        model = AutoModelForMultipleChoice.from_pretrained(base_model_name)
    elif bert_pooling == "mean":
        model = BertForMultipleChoiceWithMeanPooling(model_name=base_model_name,
                                                     model_config_path=model_config_path, )
    else:
        raise RuntimeError(f"Unsupported bert_pooling: {bert_pooling}")
    if freeze_embs:
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False

    if freeze_layers:
        for layer in model.bert.encoder.layer[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    metric_name = "accuracy"

    args = TrainingArguments(
        output_finetuned_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        logging_steps=0.01,
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
        fp16=fp16,
        seed=42,
        # data_seed=42,
        save_total_limit=2,
        push_to_hub=False,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    # Dev evaluation
    dev_eval_dict = trainer.evaluate()
    logging.info(f"Final DEV evaluation:")
    for k, v in dev_eval_dict.items():
        logging.info(f"{k}: {v}")

    # Test evaluation
    test_eval_dict = trainer.evaluate(test_dataset)
    logging.info(f"Final DEV evaluation:")
    for k, v in test_eval_dict.items():
        logging.info(f"{k}: {v}")
    logging.info("Finished Evaluation....")

    dev_accuracy = dev_eval_dict["eval_accuracy"]
    test_accuracy = test_eval_dict["eval_accuracy"]

    print(f"final scores (dev accuracy): {dev_accuracy}")
    print(f"final scores (test accuracy): {test_accuracy}")
    logging.info(f"final scores (dev accuracy): {dev_accuracy}")
    logging.info(f"final scores (test accuracy): {test_accuracy}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', type=str, required=True)
    parser.add_argument('--base_model_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=False, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, required=False, default=1)
    parser.add_argument('--max_length', type=int, required=False, default=512)
    parser.add_argument('--bert_pooling', type=str, required=False, choices=("cls", "mean"), default="cls")
    parser.add_argument('--model_config_path', type=str, required=False)
    parser.add_argument('--num_epochs', type=int, required=False, default=50)
    parser.add_argument('--warmup_ratio', type=float, required=False, default=0.1)
    parser.add_argument('--warmup_steps', type=int, required=False, default=0)
    parser.add_argument('--learning_rate', type=float, required=False, default=1e-5)
    parser.add_argument("--freeze_embs", action="store_true")
    parser.add_argument("--freeze_layers", required=False, type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_5_options", action="store_true")
    parser.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()
    main(args)
