import argparse
import json
import logging
import os
import statistics

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, BertConfig
from transformers import AutoTokenizer

from textkb.bert_modeling.multiple_choice_models import BertForMultipleChoiceWithMeanPooling
from textkb.utils.io import create_dir_if_not_exists


# from datasets import load_metric


class QuestionMultipleChoiceDataset(Dataset):

    def __init__(self, data_json, max_length, tokenizer,
                 answer_choice_fields, question_field, answer_field, debug=False):
        super(QuestionMultipleChoiceDataset).__init__()

        # self.answer_choice_letters = ('A', 'B', 'C', 'D', 'E')
        self.answer_choice_fields = answer_choice_fields
        self.question_field = question_field
        self.answer_field = answer_field
        self.choice2id = {ch: i for i, ch in enumerate(self.answer_choice_fields)}
        self.tokenizer = tokenizer
        self.sep_token = tokenizer.sep_token
        self.max_length = max_length
        self.tokenized_paired_samples = []
        labels = []

        for i, sample_dict in enumerate(data_json):
            question = sample_dict[self.question_field]
            options = tuple(map(lambda x: sample_dict[x],
                                self.answer_choice_fields))

            samples = [[question, op] for op in options]
            tokenized_sample = tokenizer.batch_encode_plus(samples,
                                                           max_length=self.max_length,
                                                           padding="max_length",
                                                           truncation=True,
                                                           return_tensors="pt")
            if debug and i < 5:
                option_tokens = [tokenizer.convert_ids_to_tokens(t, skip_special_tokens=True)
                                 for t in tokenized_sample["input_ids"]]
                # node_tokens = tokenizer.convert_ids_to_tokens(concept_input_ids)
                option_strs = ["".join((x.strip("#") if x.startswith("#") else f" {x}" for x in t))
                               for t in option_tokens]
                print("\n".join(option_strs), '\n--')

            assert tokenized_sample["input_ids"].size() == (len(options), self.max_length)
            self.tokenized_paired_samples.append(tokenized_sample)
            labels.append(sample_dict[self.answer_field])

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
    for data_dict in json_list:
        assert isinstance(data_dict["cop"], int)
        data_dict["cop"] = data_dict["cop"] - 1

    return np.array(json_list)


def calculate_actual_max_length(data_dict_list, tokenizer, question_field, answer_choice_fields, max_length):
    sequences = []
    for data_dict in data_dict_list:
        question = data_dict[question_field]
        options = tuple(map(lambda x: data_dict[x], answer_choice_fields))
        samples = [[question, op] for op in options]
        sequences.extend(samples)
    input_ids = tokenizer.batch_encode_plus(sequences,
                                            max_length=max_length,
                                            truncation=True)["input_ids"]
    act_max_length = max((len(t) for t in input_ids))

    return act_max_length


def main(args):
    input_data_dir = args.input_data_dir
    max_length = args.max_length
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    bert_pooling = args.bert_pooling
    model_config_path = args.model_config_path
    warmup_ratio = args.warmup_ratio
    warmup_steps = args.warmup_steps
    base_model_name = args.base_model_name
    freeze_embs = args.freeze_embs
    freeze_layers = args.freeze_layers
    num_folds = args.num_folds
    fp16 = args.fp16
    debug = args.debug
    output_dir = args.output_dir
    output_finetuned_dir = os.path.join(output_dir, "finetuned_models/")
    create_dir_if_not_exists(output_finetuned_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    question_field = args.question_field
    answer_field = args.answer_field
    answer_choice_fields = args.option_fields

    train_json_path = os.path.join(input_data_dir, "train.json")
    dev_json_path = os.path.join(input_data_dir, "dev.json")
    # test_json_path = os.path.join(input_data_dir, "test.json")

    train_dev_data = load_jsonl_data(train_json_path)
    actual_train_dev_max_length = calculate_actual_max_length(data_dict_list=train_dev_data,
                                                              tokenizer=tokenizer,
                                                              question_field=question_field,
                                                              answer_choice_fields=answer_choice_fields,
                                                              max_length=max_length)
    print(f"Actual train+dev max length: {actual_train_dev_max_length}")
    test_data = load_jsonl_data(dev_json_path)
    actual_test_max_length = calculate_actual_max_length(data_dict_list=test_data,
                                                         tokenizer=tokenizer,
                                                         question_field=question_field,
                                                         answer_choice_fields=answer_choice_fields,
                                                         max_length=max_length)
    print(f"Actual test max length: {actual_test_max_length}")
    # test_data = load_jsonl_data(test_json_path)
    logging.info(f"Loaded data. Train + dev - {len(train_dev_data)}, Test - {len(test_data)}, "
                 f"Total - {len(train_dev_data) + len(test_data)}")

    test_dataset = QuestionMultipleChoiceDataset(test_data, max_length=actual_test_max_length,
                                                 tokenizer=tokenizer,
                                                 question_field=question_field,
                                                 answer_field=answer_field,
                                                 answer_choice_fields=answer_choice_fields,
                                                 debug=debug)
    kfold = KFold(n_splits=num_folds, random_state=42, shuffle=True)
    dev_accuracies_list = []
    test_accuracies_list = []
    for fold_id, (train_index, test_index) in enumerate(kfold.split(train_dev_data)):
        train_data = train_dev_data[train_index]
        dev_data = train_dev_data[test_index]

        train_dataset = QuestionMultipleChoiceDataset(train_data, max_length=actual_train_dev_max_length,
                                                      tokenizer=tokenizer,
                                                      question_field=question_field,
                                                      answer_field=answer_field,
                                                      answer_choice_fields=answer_choice_fields,
                                                      debug=debug)
        dev_dataset = QuestionMultipleChoiceDataset(dev_data, max_length=actual_train_dev_max_length,
                                                    tokenizer=tokenizer,
                                                    question_field=question_field,
                                                    answer_field=answer_field,
                                                    answer_choice_fields=answer_choice_fields,
                                                    debug=debug)

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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', )
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
    parser.add_argument("--num_folds", required=False, type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--question_field", type=str, required=False,
                        default="question")
    parser.add_argument("--option_fields", type=str, nargs='+', required=False,
                        default=('opa', 'opb', 'opc', 'opd'))
    parser.add_argument("--answer_field", type=str, required=False,
                        default="cop")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()
    main(args)
