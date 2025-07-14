import argparse
import json
import logging
import os

from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
import statistics

from textkb.bert_modeling.sequence_classification_models import BertForSequenceClassificationWithMeanPooling
# from datasets import load_metric

from textkb.utils.io import create_dir_if_not_exists

# metric = load_metric('accuracy')
import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    labels = np.argmax(labels, axis=1)
    return {"accuracy": float(
        accuracy_score(labels, predictions, normalize=True, sample_weight=None))}

    # return metric.compute(predictions=predictions, references=labels)


class QuestionContextBinaryQADataset(Dataset):

    def __init__(self, questions, contexts, labels, max_length, tokenizer):
        super(QuestionContextBinaryQADataset).__init__()

        self.questions = questions
        self.contexts = contexts
        self.max_length = max_length
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer = tokenizer

        self.tokenized_input = [tokenizer.encode_plus(x, y,
                                                      max_length=self.max_length,
                                                      padding="max_length",
                                                      truncation=True,
                                                      # truncation='only_second',
                                                      return_tensors="pt", ) \
                                for x, y in zip(self.questions,
                                                self.contexts)]
        assert len(self.questions) == len(self.contexts) == len(self.labels)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokenized_input[idx]["input_ids"][0],
            "attention_mask": self.tokenized_input[idx]["attention_mask"][0],
            "labels": self.labels[idx]}


LABEL_MAP = {
    "yes": 0,
    "no": 1,
    "maybe": 2,
}


def dataset_json2lists(json_data, context_field="CONTEXTS"):
    questions = []
    contexts = []
    labels = []
    for pmid, data_dict in json_data.items():
        question = data_dict["QUESTION"]
        if context_field == "CONTEXTS":
            context = " ".join(data_dict["CONTEXTS"])
        else:
            raise RuntimeError(f"Unsupported context_field: {context_field}")
        label = [0, ] * 3
        label[LABEL_MAP[data_dict["final_decision"]]] = 1
        questions.append(question)
        contexts.append(context)
        labels.append(label)
    return questions, contexts, labels


def main(args):
    input_data_dir = args.input_data_dir
    num_folds = args.num_folds
    max_length = args.max_length
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    bert_pooling = args.bert_pooling
    model_config_path = args.model_config_path
    warmup_ratio = args.warmup_ratio
    warmup_steps = args.warmup_steps
    fp16 = args.fp16
    base_model_name = args.base_model_name
    freeze_embs = args.freeze_embs
    freeze_layers = args.freeze_layers
    output_dir = args.output_dir
    output_finetuned_dir = os.path.join(output_dir, "finetuned_models/")
    create_dir_if_not_exists(output_finetuned_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    test_json_path = os.path.join(input_data_dir, "test_set.json")
    test_data = json.load(open(test_json_path))
    test_q, test_c, test_labels = dataset_json2lists(test_data)
    test_dataset = QuestionContextBinaryQADataset(questions=test_q, contexts=test_c, labels=test_labels,
                                                  max_length=max_length, tokenizer=tokenizer)

    dev_accuracies_list = []
    test_accuracies_list = []
    for fold_id in range(num_folds):
        logging.info(f"Processing fold {fold_id}")
        fold_dir = os.path.join(input_data_dir, f"pqal_fold{fold_id}/")
        train_json_path = os.path.join(fold_dir, "train_set.json")
        dev_json_path = os.path.join(fold_dir, "dev_set.json")

        train_data = json.load(open(train_json_path))
        dev_data = json.load(open(dev_json_path))

        train_q, train_c, train_labels = dataset_json2lists(train_data)
        dev_q, dev_c, dev_labels = dataset_json2lists(dev_data)
        pass
        # model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=3)
        if bert_pooling == "cls":
            model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=3)
        elif bert_pooling == "mean":
            model = BertForSequenceClassificationWithMeanPooling(model_name=base_model_name,
                                                                 model_config_path=model_config_path,
                                                                 num_labels=3)
        else:
            raise RuntimeError(f"Unsupported bert_pooling: {bert_pooling}")
        if freeze_embs:
            for param in model.bert.embeddings.parameters():
                param.requires_grad = False

        if freeze_layers:
            for layer in model.bert.encoder.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        train_dataset = QuestionContextBinaryQADataset(questions=train_q, contexts=train_c, labels=train_labels,
                                                       max_length=max_length, tokenizer=tokenizer)

        dev_dataset = QuestionContextBinaryQADataset(questions=dev_q, contexts=dev_c, labels=dev_labels,
                                                     max_length=max_length, tokenizer=tokenizer)
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
            # data_seed=42,
            push_to_hub=False,
        )

        trainer = Trainer(
            model,
            train_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
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
    parser.add_argument("--freeze_layers", required=False, type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()
    main(args)
