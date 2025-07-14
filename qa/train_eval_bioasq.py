import argparse
import json
import logging
import os
import statistics

# metric = load_metric('accuracy')
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from transformers import AutoTokenizer

from textkb.bert_modeling.sequence_classification_models import BertForSequenceClassificationWithMeanPooling
from textkb.utils.io import create_dir_if_not_exists


# from datasets import load_metric


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    labels = np.argmax(labels, axis=1)
    return {"accuracy": float(
        accuracy_score(labels, predictions, normalize=True, sample_weight=None))}


class BinaryQADataset(Dataset):

    def __init__(self, questions, labels, max_length, tokenizer):
        super(BinaryQADataset).__init__()

        self.questions = questions
        self.max_length = max_length
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.tokenized_input = [tokenizer.encode_plus(x,
                                                      max_length=self.max_length,
                                                      truncation=True,
                                                      return_tensors="pt", ) for x in self.questions]
        assert len(self.questions) == len(self.labels)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokenized_input[idx]["input_ids"][0],
            "attention_mask": self.tokenized_input[idx]["attention_mask"][0],
            "labels": self.labels[idx]}


LABEL2ID = {
    "yes": 1,
    "no": 0,
}


def dataset_json2lists(json_data):
    samples = []
    for q_dict in json_data["questions"]:
        q_type = q_dict["type"]
        if q_type != "yesno":
            continue
        label = [0, ] * 2
        label[LABEL2ID[q_dict["exact_answer"]]] = 1
        q_text = q_dict["body"]

        samples.append((q_text, label))

    return samples


def main(args):
    input_data_dir = args.input_data_dir
    num_folds = args.num_folds
    max_length = args.max_length
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    bert_pooling = args.bert_pooling
    warmup_ratio = args.warmup_ratio
    warmup_steps = args.warmup_steps
    base_model_name = args.base_model_name
    freeze_embs = args.freeze_embs
    freeze_layers = args.freeze_layers
    fp16 = args.fp16
    model_config_path = args.model_config_path
    output_dir = args.output_dir
    output_finetuned_dir = os.path.join(output_dir, "finetuned_models/")
    create_dir_if_not_exists(output_finetuned_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    input_train_dev_path = os.path.join(input_data_dir, "BioASQ-training11b/training11b.json")
    train_dev_json_data = json.load(open(input_train_dev_path))

    samples = dataset_json2lists(train_dev_json_data)
    train_dev_q, train_dev_labels = np.array([t[0] for t in samples]), np.array([t[1] for t in samples])

    test_q, test_labels = [], []
    test_part_ids = (1, 2, 3, 4)
    for tpid in test_part_ids:
        test_part_path = os.path.join(input_data_dir,
                                      f"Task11BGoldenEnriched/11B{tpid}_golden.json")
        test_part_data = json.load(open(test_part_path))

        ts = dataset_json2lists(test_part_data)
        tq = [t[0] for t in ts]
        tl = [t[1] for t in ts]

        test_q.extend(tq)
        test_labels.extend(tl)
    test_dataset = BinaryQADataset(questions=test_q, labels=test_labels, max_length=max_length,
                                   tokenizer=tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    kfold = KFold(n_splits=num_folds, random_state=42, shuffle=True)
    dev_accuracies_list = []
    test_accuracies_list = []
    for fold_id, (train_index, test_index) in enumerate(kfold.split(samples)):
        logging.info(f"Processing fold {fold_id}")
        train_q = train_dev_q[train_index]
        train_labels = train_dev_labels[train_index]
        dev_q = train_dev_q[test_index]
        dev_labels = train_dev_labels[test_index]

        # train_samples = samples[train_index]
        # dev_samples = samples[test_index]
        #
        # train_q = [t[0] for t in train_samples]
        # train_labels = [t[1] for t in train_samples]
        # dev_q = [t[0] for t in dev_samples]
        # dev_labels = [t[1] for t in dev_samples]

        train_dataset = BinaryQADataset(questions=train_q, labels=train_labels, max_length=max_length,
                                        tokenizer=tokenizer)

        dev_dataset = BinaryQADataset(questions=dev_q, labels=dev_labels, max_length=max_length,
                                      tokenizer=tokenizer)

        if bert_pooling == "cls":
            model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
        elif bert_pooling == "mean":
            # model_name, model_config_path
            model = BertForSequenceClassificationWithMeanPooling(model_name=base_model_name,
                                                                 model_config_path=model_config_path,
                                                                 num_labels=2)
        else:
            raise RuntimeError(f"Unsupported bert_pooling: {bert_pooling}")
        if freeze_embs:
            for param in model.bert.embeddings.parameters():
                param.requires_grad = False

        if freeze_layers:
            for layer in model.bert.encoder.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        # model = AutoModelForSequenceClassification.from_pretrained(base_model_name,
        #                                                            num_labels=2)

        metric_name = "accuracy"
        train_args = TrainingArguments(
            output_finetuned_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            warmup_ratio=warmup_ratio,
            warmup_steps=warmup_steps,
            fp16=fp16,
            load_best_model_at_end=True,
            metric_for_best_model=metric_name,
            logging_steps=0.01,
            save_total_limit=2,
            seed=42,
            push_to_hub=False,
        )

        trainer = Trainer(
            model,
            train_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        logging.info("Training...")
        trainer.train()
        logging.info("Finished training. Evaluating....")
        dev_eval_dict = trainer.evaluate()
        logging.info(f"Fold {fold_id} dev:")
        print(f"Fold {fold_id} dev:")
        for k, v in dev_eval_dict.items():
            print(f"{k}: {v}")
            logging.info(f"{k}: {v}")

        test_eval_dict = trainer.evaluate(test_dataset)
        print(f"Fold {fold_id} test:")
        logging.info(f"Fold {fold_id} test:")
        for k, v in test_eval_dict.items():
            print(f"{k}: {v}")
            logging.info(f"{k}: {v}")
        logging.info("Finished Evaluation....")

        dev_accuracies_list.append(dev_eval_dict["eval_accuracy"])
        test_accuracies_list.append(test_eval_dict["eval_accuracy"])

    overall_dev_acc = sum(dev_accuracies_list) / len(dev_accuracies_list)
    overall_test_acc = sum(test_accuracies_list) / len(test_accuracies_list)
    overall_dev_std = statistics.stdev(dev_accuracies_list)
    overall_test_std = statistics.stdev(test_accuracies_list)

    print(f"final scores (mean dev accuracy): {overall_dev_acc}")
    print(f"final scores (std dev accuracy): {overall_dev_std}")
    print(f"final scores (individual dev accuracies): {','.join((str(x) for x in dev_accuracies_list))}")

    print(f"final scores (mean test accuracy): {overall_test_acc}")
    print(f"final scores (std test accuracy): {overall_test_std}")
    print(f"final scores (individual test accuracies): {','.join((str(x) for x in test_accuracies_list))}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', type=str, required=True)
    parser.add_argument('--base_model_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=False, default=16)
    parser.add_argument('--max_length', type=int, required=False, default=512)
    parser.add_argument('--gradient_accumulation_steps', type=int, required=False, default=1)
    parser.add_argument('--bert_pooling', type=str, required=False, choices=("cls", "mean"), default="cls")
    parser.add_argument('--model_config_path', type=str, required=False)
    parser.add_argument('--num_epochs', type=int, required=False, default=50)
    parser.add_argument('--warmup_ratio', type=float, required=False, default=0.1)
    parser.add_argument('--warmup_steps', type=int, required=False, default=0)
    parser.add_argument('--learning_rate', type=float, required=False, default=1e-5)
    parser.add_argument('--num_folds', type=int, required=False, default=10)
    parser.add_argument("--freeze_embs", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--freeze_layers", required=False, type=int)
    parser.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()
    main(args)
