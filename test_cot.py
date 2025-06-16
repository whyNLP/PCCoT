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
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from itertools import chain
from typing import Optional

import datasets
import evaluate
import torch
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_xla_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry, cached_file
from transformers.utils.versions import require_version
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, AutoPeftModel
from peft.utils import CONFIG_NAME as PEFT_CONFIG_NAME

import models


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.48.0.dev0")

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class COTDataProcessor:
    """
    Data processor for the COT model.

    Example:
    {"question": "Janets ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?", "steps": ["<<16-3-4=9>>", "<<9*2=18>>"], "answer": "18"}
    """

    def __init__(
        self,
        tokenizer: models.data_processor.PreTrainedTokenizer,
        pccot_args: models.data_processor.PCCoTArguments,
        max_seq_length: int = 8192,
    ) -> None:
        self.tokenizer = tokenizer
        self.pccot_args = pccot_args
        self.max_seq_length = max_seq_length

        self.tokenized_answer_prompt = self.tokenizer.encode(
            pccot_args.answer_prompt, add_special_tokens=False
        )

    def tokenize_function(self, examples):
        output = {
            "question": self.tokenizer(
                examples["question"],
                return_attention_mask=False,
                add_special_tokens=True,
            )["input_ids"],
            "steps": [
                self.tokenizer(
                    steps, return_attention_mask=False, add_special_tokens=False
                )["input_ids"]
                for steps in examples["steps"]
            ],
            "answer": models.data_processor.batch_tokenize_number(self.tokenizer, examples["answer"]),
        }
        return output

    def group_texts(self, examples):
        """
        We need to build cot sequence and ccot sequence.
        The cot sequence is built by concatenating the question, steps, and answer.
        The ccot sequence is built by concatenating the question, latent tokens, and answer.
        For both the sequences, we need to provide a list of indices to indicate the last token of the answer prompt.
        """
        # Building the cot sequence
        cot_input_ids = [
            question
            + [token for step in steps for token in step]
            + self.tokenized_answer_prompt
            + answer
            + [self.tokenizer.eos_token_id]
            for question, steps, answer in zip(
                examples["question"], examples["steps"], examples["answer"]
            )
        ]
        cot_labels = [
            [self.pccot_args.label_pad_token_id] * len(question)
            + [token for step in steps for token in step]
            + self.tokenized_answer_prompt
            + answer
            + [self.tokenizer.eos_token_id]
            for question, steps, answer in zip(
                examples["question"], examples["steps"], examples["answer"]
            )
        ]
        return {
            "input_ids": cot_input_ids,
            "labels": cot_labels,
        }

    def data_collator(self, features):
        """
        We need to build cot sequence and ccot sequence, with proper padding.
        The cot sequence is built by concatenating the question, steps, and answer.
        The ccot sequence is built by concatenating the question, latent tokens, and answer. Using left padding for the questions.
        For both the sequences, we need to provide a list of indices to indicate the last token of the answer prompt.
        """
        # Padding the question, steps, and answer
        cot_input_ids = self.tokenizer.pad(
            {"input_ids": [item['input_ids'] for item in features]},
            padding=True,
            return_tensors="pt",
        )
        cot_input_ids, attention_mask = cot_input_ids["input_ids"], cot_input_ids["attention_mask"]
        cot_labels = self.tokenizer.pad(
            {"input_ids": [item['labels'] for item in features]},
            padding=True,
            return_tensors="pt",
        )["input_ids"]
        cot_labels[cot_labels == self.tokenizer.pad_token_id] = self.pccot_args.label_pad_token_id

        return {
            "input_ids": cot_input_ids,
            "labels": cot_labels,
            "attention_mask": attention_mask,
        }


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
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
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    # Attention Implementation arguments
    attn_implementation: str = field(
        default="eager",
        metadata={
            "help": (
                "The implementation of the attention layer. "
                "This is useful for models that have special tokens at the beginning of the input."
            ),
            "choices": ["eager", "flash_attention_2", "sdpa"]
        },
    )

    def __post_init__(self):
        # if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
        #     raise ValueError(
        #         "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
        #     )

        pass


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`test_file` should be a csv, a json or a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, models.PCCoTArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, pccot_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, pccot_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
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
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
            trust_remote_code=model_args.trust_remote_code,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
                trust_remote_code=model_args.trust_remote_code,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
                trust_remote_code=model_args.trust_remote_code,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    # allow config overrides under all circumstances
    if model_args.config_overrides is not None:
        logger.info(f"Overriding config: {model_args.config_overrides}")
        config.update_from_string(model_args.config_overrides)
        logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    # >>> add <bot>, <latent> and <eot> to the tokenizer
    # pcot: parallel chain-of-thought with continuous tokens
    def get_special_token(tokenizer, token):
        if token not in tokenizer.additional_special_tokens:
            tokenizer.add_special_tokens({'additional_special_tokens': (token, )}, replace_additional_special_tokens=False)
        return tokenizer.convert_tokens_to_ids(token)

    if pccot_args.bot_token_id is None:
        pccot_args.bot_token_id = get_special_token(tokenizer, '<pcot.bot>')
    if pccot_args.eot_token_id is None:
        pccot_args.eot_token_id = get_special_token(tokenizer, '<pcot.eot>')
    if pccot_args.latent_token_id is None:
        pccot_args.latent_token_id = get_special_token(tokenizer, '<pcot.latent>')
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({'eos_token': '[EOS]'})
    # <<<

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )

        # check whether we should load with AutoModelForCausalLM or AutoPeftModel
        if (
            cached_file(
                model_args.model_name_or_path,
                PEFT_CONFIG_NAME,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                _raise_exceptions_for_missing_entries=False,
            )
            is not None
        ):
            model = AutoPeftModel.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=model_args.low_cpu_mem_usage,
                attn_implementation=model_args.attn_implementation
            )

            # we have to override the model config after loading the model, peft does not provide interface to
            # load base model with custom config with AutoPeftModel.
            model.get_base_model().config = config

        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=model_args.low_cpu_mem_usage,
                attn_implementation=model_args.attn_implementation,
            )
    else:
        model = AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=model_args.attn_implementation,
        )
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    if hasattr(config, "max_position_embeddings"):
        max_pos_embeddings = config.max_position_embeddings
    else:
        # Define a default value if the attribute is missing in the config.
        max_pos_embeddings = 1024

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > max_pos_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, max_pos_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            if max_pos_embeddings > 0:
                block_size = min(1024, max_pos_embeddings)
            else:
                block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    data_processor = COTDataProcessor(
        tokenizer=tokenizer,
        pccot_args=pccot_args,
        max_seq_length=block_size,
    )

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                data_processor.tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
                # cache_file_names={"train": "/tmp/dataset_cache_train.arrow", "validation": "/tmp/dataset_cache_valid.arrow"},
            )
        else:
            tokenized_datasets = raw_datasets.map(
                data_processor.tokenize_function,
                batched=True,
                remove_columns=column_names,
            )

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            lm_datasets = tokenized_datasets.map(
                data_processor.group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
                # cache_file_names={"train": "/tmp/dataset_cache_train2.arrow", "validation": "/tmp/dataset_cache_valid2.arrow"},
            )
        else:
            lm_datasets = tokenized_datasets.map(
                data_processor.group_texts,
                batched=True,
            )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            # ccot_logits, cot_logits = logits
            # return ccot_logits.argmax(dim=-1), cot_logits.argmax(dim=-1)
            return logits.argmax(dim=-1)

        metric = evaluate.load("exact_match", cache_dir=model_args.cache_dir)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds

            # ignore tokens after eos_token_id
            def ignore_after_eos(tokens):
                tokens = tokens.tolist()
                if tokenizer.eos_token_id in tokens:
                    tokens = tokens[:tokens.index(tokenizer.eos_token_id)]
                return tokens

            # cot preds
            preds[preds == -100] = tokenizer.pad_token_id
            preds[labels == -100] = tokenizer.pad_token_id
            preds = [ignore_after_eos(pred) for pred in preds]
            decoded_cot_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_cot_preds = [
                # only keep the string after pccot_args.answer_prompt
                pred[pred.index(pccot_args.answer_prompt) + len(pccot_args.answer_prompt):] if pccot_args.answer_prompt in pred else pred
                for pred in decoded_cot_preds
            ]
            labels[labels == -100] = tokenizer.pad_token_id
            labels = [ignore_after_eos(label) for label in labels]
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            decoded_cot_labels = [
                # only keep the string after pccot_args.answer_prompt
                label[label.index(pccot_args.answer_prompt) + len(pccot_args.answer_prompt):] if pccot_args.answer_prompt in label else label
                for label in decoded_labels
            ]

            logger.info("CoT Results")
            for i, pred, label in zip(range(10), decoded_cot_preds, decoded_cot_labels):
                logger.info(f"pred  {i}: {pred}")
                logger.info(f"label {i}: {label}")

            cot_result = metric.compute(predictions=decoded_cot_preds, references=decoded_cot_labels)

            return {
                "cot_exact_match": cot_result["exact_match"],
            }

    if training_args.do_predict:
        if "test" not in tokenized_datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = lm_datasets["test"]

    pccot_args.save(training_args.output_dir)

    if pccot_args.use_peft and not isinstance(model, PeftModel):
        is_gpt2 = "gpt2" in config.model_type
        embedding_layer_name = "wte" if is_gpt2 else "embed_tokens"
        special_token_ids = [pccot_args.bot_token_id, pccot_args.eot_token_id, pccot_args.latent_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id]
        peft_config = LoraConfig(
            inference_mode=False, r=pccot_args.lora_r, lora_alpha=pccot_args.lora_alpha, lora_dropout=pccot_args.lora_dropout,
            target_modules=pccot_args.lora_target_modules.split("-"),
            trainable_token_indices={embedding_layer_name: special_token_ids},
            modules_to_save=pccot_args.lora_modules_to_save.split("-") if pccot_args.lora_modules_to_save else None,
            fan_in_fan_out=is_gpt2,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=data_processor.data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_xla_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_xla_available()
        else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        # if this is a peft model, we need to manually save the config
        if isinstance(model, PeftModel):
            trainer.model.config.save_pretrained(training_args.output_dir)

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    def eval_cot(model, split = 'test'):
        from tqdm import tqdm
        from itertools import batched

        if not model.generation_config.pad_token_id:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        metric = evaluate.load("exact_match", cache_dir=model_args.cache_dir)
        preds = []
        labels = []

        for batch in tqdm(batched(raw_datasets[split], n=training_args.per_device_eval_batch_size), desc=f"Evaluating {split} set", total=len(raw_datasets[split])//training_args.per_device_eval_batch_size):
            questions = [item['question'] for item in batch]
            inputs = tokenizer(questions, padding=True, padding_side='left', add_special_tokens=True, return_tensors="pt").to(model.device)

            # gpt-2 uses absolute positional encoding, so we need to setup the position_ids
            position_ids = inputs['attention_mask'].cumsum(dim=-1) - 1
            inputs['position_ids'] = position_ids.masked_fill(inputs['attention_mask'] == 0, 0)

            # 1) greedy decoding
            outputs = model.generate(**inputs, do_sample=False, max_length=1024)

            # # 2) or sampling from the distribution
            # outputs = model.generate(**inputs, do_sample=True, max_length=1024,
            #     top_k=50,
            #     temperature=0.1,
            # )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # extract the answer from the decoded text
            for pred, gold in zip(decoded, batch):
                if pccot_args.answer_prompt in pred:
                    pred = pred.split(pccot_args.answer_prompt)[-1]
                preds.append(pred)
                labels.append(gold['answer'])
        
        cot_result = metric.compute(predictions=preds, references=labels)
        return cot_result


    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = eval_cot(trainer.model, "validation")

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Evaluation
    if training_args.do_predict:
        logger.info("*** Predict ***")

        metrics = eval_cot(trainer.model, "test")

        metrics["test_samples"] = len(test_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()