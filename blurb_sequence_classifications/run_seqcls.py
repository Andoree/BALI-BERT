#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification.

Adapted from
https://github.com/huggingface/transformers/blob/72aee83ced5f31302c5e331d896412737287f976/examples/pytorch/text-classification/run_glue.py
"""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import statistics
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed, requires_backends,
)
from transformers.data.metrics import DEPRECATION_WARNING
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import numpy as np

LABELS = ['activating invasion and metastasis', 'avoiding immune destruction',
          'cellular energetics', 'enabling replicative immortality', 'evading growth suppressors',
          'genomic instability and mutation', 'inducing angiogenesis', 'resisting cell death',
          'sustaining proliferative signaling', 'tumor promoting inflammation']


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def divide(x, y):
    return np.true_divide(x, y, out=np.zeros_like(x, dtype=np.float32), where=y != 0)


def compute_p_r_f(preds, labels):
    TP = ((preds == labels) & (preds != 0)).astype(int).sum()
    P_total = (preds != 0).astype(int).sum()
    L_total = (labels != 0).astype(int).sum()
    P = divide(TP, P_total).mean()
    R = divide(TP, L_total).mean()
    F1 = divide(2 * P * R, (P + R)).mean()
    return P, R, F1


def eval_hoc(true_list, pred_list, id_list):
    data = {}

    assert len(true_list) == len(pred_list) == len(id_list), \
        f'Gold line no {len(true_list)} vs Prediction line no {len(pred_list)} vs Id line no {len(id_list)}'

    cat = len(LABELS)
    assert cat == len(true_list[0]) == len(pred_list[0])

    for i in range(len(true_list)):
        id = id_list[i]
        key = id.split('_')[0]
        if key not in data:
            data[key] = (set(), set())

        for j in range(cat):
            if true_list[i][j] == 1:
                data[key][0].add(j)
            if pred_list[i][j] == 1:
                data[key][1].add(j)

    print(f"There are {len(data)} documents in the data set")
    # print ('data', data)

    y_test = []
    y_pred = []
    for k, (true, pred) in data.items():
        t = [0] * len(LABELS)
        for i in true:
            t[i] = 1

        p = [0] * len(LABELS)
        for i in pred:
            p[i] = 1

        y_test.append(t)
        y_pred.append(p)

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    p, r, f1 = compute_p_r_f(y_pred, y_test)
    return {"precision": p, "recall": r, "F1": f1}


from transformers import Trainer, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


class SeqClsTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=None,
            ignore_keys=ignore_keys,
        )
        # self.label_names = label_names
        self.compute_metrics = compute_metrics

        # metrics = output.metrics
        metrics = self.compute_metrics(output, eval_dataset)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        self.log(metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, predict_dataset, ignore_keys=None, metric_key_prefix: str = "test"):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.evaluation_loop
        output = eval_loop(
            predict_dataloader,
            description="Prediction",
            prediction_loss_only=None,
            ignore_keys=ignore_keys,
        )

        # self.label_names = label_names
        self.compute_metrics = compute_metrics

        # metrics = output.metrics
        metrics = self.compute_metrics(output, predict_dataset)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        self.log(metrics)  # Added

        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=metrics)


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.9.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    metric_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the metric"},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    initial_seed: int = field(
        default=0
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    num_folds: int = field(
        default=3,
        metadata={
            "help": "Number of Cross-validation folds"},
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                    validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_embs: bool = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_layers: Optional[int] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

    num_folds = data_args.num_folds
    test_evaluation_dictionary = {}
    dev_evaluation_dictionary = {}
    for fold_id in range(num_folds):

        logger.info(f"Training/evaluation parameters {training_args}")

        # Detecting last checkpoint.
        last_checkpoint = None
        if os.path.isdir(
                training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

        # Set seed before initializing model.
        set_seed(fold_id + data_args.initial_seed)

        # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
        # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
        #
        # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
        # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
        # label if at least two columns are provided.
        #
        # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
        # single column. You can easily tweak this behavior (see below)
        #
        # In distributed training, the load_dataset function guarantee that only one local process can concurrently
        # download the dataset.
        if data_args.task_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)
        elif data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
            )
        else:
            # Loading a dataset from your local files.
            # CSV/JSON training and evaluation files are needed.
            data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

            # Get the test dataset: you can provide your own CSV/JSON test file (see below)
            # when you use `do_predict` without specifying a GLUE benchmark task.
            if training_args.do_predict:
                if data_args.test_file is not None:
                    train_extension = data_args.train_file.split(".")[-1]
                    test_extension = data_args.test_file.split(".")[-1]
                    assert (
                            test_extension == train_extension
                    ), "`test_file` should have the same extension (csv or json) as `train_file`."
                    data_files["test"] = data_args.test_file
                else:
                    raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

            for key in data_files.keys():
                logger.info(f"load a local file for {key}: {data_files[key]}")

            if data_args.train_file.endswith(".csv"):
                # Loading a dataset from local csv files
                raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
            else:
                # Loading a dataset from local json files
                raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
        # See more about loading any type of standard or custom dataset at
        # https://huggingface.co/docs/datasets/loading_datasets.html.

        # Labels
        if data_args.task_name is not None:
            is_regression = data_args.task_name == "stsb"
            if not is_regression:
                label_list = raw_datasets["train"].features["label"].names
                num_labels = len(label_list)
            else:
                num_labels = 1
        else:
            # Trying to have good defaults here, don't hesitate to tweak to your needs.
            is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
            is_multiclass_binary = raw_datasets["train"].features["label"].dtype in ["list"]
            if is_regression:
                print('is_regression')
                num_labels = 1
            elif is_multiclass_binary:
                print('is_multiclass_binary')
                assert data_args.metric_name.startswith("hoc")
                num_labels = len(raw_datasets["train"][0]["label"])
                label_list = list(range(num_labels))
            else:
                # A useful fast method:
                # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
                label_list = raw_datasets["train"].unique("label")
                label_list.sort()  # Let's sort it for determinism
                print('\nlabel_list', label_list)
                num_labels = len(label_list)

        # Load pretrained model and tokenizer
        #
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model_class = AutoModelForSequenceClassification
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model_args
        if model_args.freeze_embs:
            for param in model.bert.embeddings.parameters():
                param.requires_grad = False

        if model_args.freeze_layers:
            for layer in model.bert.encoder.layer[:model_args.freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        # Preprocessing the raw_datasets
        if data_args.task_name is not None:
            sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
        else:
            # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
            non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
            if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
                sentence1_key, sentence2_key = "sentence1", "sentence2"
            elif "sentence" in non_label_column_names:
                sentence1_key, sentence2_key = "sentence", None
            else:
                if len(non_label_column_names) >= 2:
                    sentence1_key, sentence2_key = non_label_column_names[:2]
                else:
                    sentence1_key, sentence2_key = non_label_column_names[0], None

        # Padding strategy
        if data_args.pad_to_max_length:
            padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            padding = False

        # Some models have set the order of the labels to use, so let's make sure we do use it.
        label_to_id = None
        if (
                model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
                and data_args.task_name is not None
                and not is_regression
        ):
            # Some have all caps in their config, some don't.
            label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
            if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
                label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
            else:
                logger.warning(
                    "Your model seems to have been trained with labels, but they don't match the dataset: ",
                    f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                    "\nIgnoring the model labels as a result.",
                )
        elif data_args.task_name is None and not is_regression:
            label_to_id = {v: i for i, v in enumerate(label_list)}

        if label_to_id is not None:
            model.config.label2id = label_to_id
            model.config.id2label = {id: label for label, id in config.label2id.items()}

        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        def preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key], examples[sentence2_key])
            )

            result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

            # Map labels to IDs (not necessary for GLUE tasks)
            if label_to_id is not None and "label" in examples:
                if is_multiclass_binary:
                    result["label"] = examples["label"]
                else:
                    result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
            return result

        with training_args.main_process_first(desc="dataset map pre-processing"):
            raw_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        if training_args.do_train:
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = raw_datasets["train"]
            if data_args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(data_args.max_train_samples))

        if training_args.do_eval:
            if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
            if data_args.max_eval_samples is not None:
                eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

        if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
            if "test" not in raw_datasets and "test_matched" not in raw_datasets:
                raise ValueError("--do_predict requires a test dataset")
            predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
            if data_args.max_predict_samples is not None:
                predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

        # Log a few random samples from the training set:
        # if training_args.do_train:
        #     for index in random.sample(range(len(train_dataset)), 3):
        #         logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

        # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
        # predictions and label_ids field) and has to return a dictionary string to float.
        def compute_metrics(p: EvalPrediction, eval_dataset):
            # Get the metric function
            if data_args.task_name is not None:
                metric = load_metric("glue", data_args.task_name)
            else:
                # metric = load_metric("accuracy")
                metric = simple_accuracy
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            if data_args.metric_name == "hoc":
                # from utils_hoc import eval_hoc
                labels = np.array(p.label_ids).astype(int)  # [num_ex, num_class]
                preds = (np.array(preds) > 0).astype(int)  # [num_ex, num_class]
                ids = eval_dataset["id"]
                return eval_hoc(labels.tolist(), preds.tolist(), list(ids))

            preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
            if data_args.task_name is not None:
                result = metric.compute(predictions=preds, references=p.label_ids)
                if len(result) > 1:
                    result["combined_score"] = np.mean(list(result.values())).item()
                return result
            elif data_args.metric_name == "pearsonr":
                from scipy.stats import pearsonr as scipy_pearsonr
                pearsonr = float(scipy_pearsonr(p.label_ids, preds)[0])
                return {"pearsonr": pearsonr}
            elif data_args.metric_name == "PRF1":
                TP = ((preds == p.label_ids) & (preds != 0)).astype(int).sum().item()
                P_total = (preds != 0).astype(int).sum().item()
                L_total = (p.label_ids != 0).astype(int).sum().item()
                P = TP / P_total if P_total else 0
                R = TP / L_total if L_total else 0
                F1 = 2 * P * R / (P + R) if (P + R) else 0
                return {"precision": P, "recall": R, "F1": F1}
            elif is_regression:
                return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
            else:
                return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        if data_args.pad_to_max_length:
            data_collator = default_data_collator
        elif training_args.fp16:
            data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        else:
            data_collator = None

        # Initialize our Trainer
        trainer = SeqClsTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

            trainer.save_model()  # Saves the tokenizer too for easy upload

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")

            # Loop to handle MNLI double evaluation (matched, mis-matched)
            tasks = [data_args.task_name]
            eval_datasets = [eval_dataset]
            if data_args.task_name == "mnli":
                tasks.append("mnli-mm")
                eval_datasets.append(raw_datasets["validation_mismatched"])

            for eval_dataset, task in zip(eval_datasets, tasks):
                metrics = trainer.evaluate(eval_dataset=eval_dataset)

                max_eval_samples = (
                    data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
                )
                metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if os.environ.get('USE_CODALAB', 0):
                import json
                json.dump(metrics, open("dev_stats.json", "w"))

        if training_args.do_predict:
            logger.info("*** Predict ***")

            # Loop to handle MNLI double evaluation (matched, mis-matched)
            tasks = [data_args.task_name]
            predict_datasets = [predict_dataset]
            if data_args.task_name == "mnli":
                tasks.append("mnli-mm")
                predict_datasets.append(raw_datasets["test_mismatched"])

            for predict_dataset, task in zip(predict_datasets, tasks):
                results = trainer.predict(predict_dataset, metric_key_prefix="test")
                predictions = results.predictions
                metrics = results.metrics
                metrics["test_samples"] = len(predict_dataset)

                test_eval_dict = trainer.evaluate(predict_dataset, metric_key_prefix="test")
                print(f"Fold {fold_id} test:")
                logging.info(f"Fold {fold_id} test:")
                for metric_name, metric_value in test_eval_dict.items():
                    if test_evaluation_dictionary.get(metric_name) is None:
                        test_evaluation_dictionary[metric_name] = []
                    test_evaluation_dictionary[metric_name].append(metric_value)
                    print(f"{metric_name}: {metric_value}")
                    logging.info(f"{metric_name}: {metric_value}")
                logging.info("Finished Evaluation....")

                dev_eval_dict = trainer.evaluate(eval_dataset, metric_key_prefix="test")
                print(f"Fold {fold_id} test:")
                logging.info(f"Fold {fold_id} test:")
                for metric_name, metric_value in dev_eval_dict.items():
                    if dev_evaluation_dictionary.get(metric_name) is None:
                        dev_evaluation_dictionary[metric_name] = []
                    dev_evaluation_dictionary[metric_name].append(metric_value)
                    print(f"{metric_name}: {metric_value}")
                    logging.info(f"{metric_name}: {metric_value}")
                logging.info("Finished Evaluation....")

    for metric_name, metric_values_list in test_evaluation_dictionary.items():
        print(f"final scores all test scores {metric_name}): {','.join((str(x) for x in metric_values_list))}")
        mean_value = sum(metric_values_list) / len(metric_values_list)
        std_value = statistics.stdev(metric_values_list)
        print(f"final scores (mean test {metric_name}): {mean_value} +- {std_value}")
    for metric_name, metric_values_list in dev_evaluation_dictionary.items():
        print(f"final scores all dev scores {metric_name}): {','.join((str(x) for x in metric_values_list))}")
        mean_value = sum(metric_values_list) / len(metric_values_list)
        std_value = statistics.stdev(metric_values_list)
        print(f"final scores (mean dev {metric_name}): {mean_value} +- {std_value}")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
