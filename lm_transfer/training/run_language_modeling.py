#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning the library models for language modeling
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
)
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

logger = get_logger(__name__)

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a language modeling task")
    parser.add_argument(
        "--experiment_config",
        type=str,
        default=None,
        help="Path to experiment config file."
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="lm_no_trainer",
        help="The name of the project to log to."
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="The name of the run to log to."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv, txt or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv, txt or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of train examples to this value if set.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker eval, truncate the number of evaluation examples to this value if set.",
    )
    parser.add_argument(
        "--eval_steps",
        type=str,
        default="epoch",
        help="The number of steps between evaluations. Set to `epoch` to evaluate at the end of each epoch.",
    )
    parser.add_argument(
        "--eval_iters",
        type=int,
        default=None,
        help="The number of iterations to run evaluation for. Defaults to `None` which means that the whole dataset "
             "will be used.",
    )
    parser.add_argument(
        "--preprocessed_dataset_path",
        type=str,
        default=None,
        help="Path to preprocessed dataset to load.",
    )
    parser.add_argument(
        "--save_preprocessed_dataset_path",
        type=str,
        default=None,
        help="Path to save preprocessed dataset to.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Adam epsilon to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    # MLM args block
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument(
        "--line_by_line",
        type=bool,
        default=False,
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--language_modeling_objective",
        type=str,
        default="mlm",
        choices=["clm", "mlm"],
        help="Language modeling objective for the model and training."
    )
    parser.add_argument(
        "--use_flash_attention_2",
        action="store_true",
        default=False,
        help="Whether to enable FlashAttention-2 for faster and more efficient attention computation."
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["float32", "float16", "bfloat16", "auto"],
        help="The dtype to use for the model. Defaults to `auto`, which loads the type from the model config."
    )

    args = parser.parse_args()

    # If provided, load experiment settings from config file.
    # Be aware that the parameters in the config overwrite the default and CLI parameters.
    if args.experiment_config is not None:
        with open(args.experiment_config) as f:
            config = json.load(f)
        for k, v in config.items():
            setattr(args, k, v)

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`train_file` should be a csv, json or txt file.")
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`validation_file` should be a csv, json or txt file.")

    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError("Need an `output_dir` to create a repo when `--push_to_hub` is passed.")

    return args


def validate_model(model, eval_dataloader, accelerator, per_device_eval_batch_size, eval_iters=None):
    """Evaluates the model on eval_dataloader and returns the evaluation loss and perplexity.
    Args:
        model (:obj:`torch.nn.Module`): The model to evaluate.
        eval_dataloader (:obj:`torch.utils.data.DataLoader`): The evaluation dataloader.
        accelerator (:obj:`accelerate.Accelerator`): The distributed training backend.
        per_device_eval_batch_size (:obj:`int`): The batch size per device.
        eval_iters (:obj:`int`, `optional`): The number of iterations to run evaluation for. Defaults to `None` which
            means that the whole dataset will be used.
    Returns:
        :obj:`tuple(torch.FloatTensor, float)`: A tuple with the evaluation loss and the perplexity.
    """
    model.eval()
    losses = []
    num_eval_steps = len(eval_dataloader)
    if eval_iters is not None:
        num_eval_steps = min(num_eval_steps, eval_iters)
    eval_progress_bar = tqdm(range(num_eval_steps), disable=not accelerator.is_local_main_process,
                             position=0, leave=True, desc="Evaluating")
    for step, batch in enumerate(eval_dataloader):
        if eval_iters is not None and step >= eval_iters:
            break

        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather_for_metrics(loss.repeat(per_device_eval_batch_size)))
        eval_progress_bar.update(1)

    losses = torch.cat(losses)
    eval_loss = torch.mean(losses)
    try:
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    return eval_loss, perplexity


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_lm_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"Set seed to {args.seed}")

    # Handle the repository creation
    api = None
    repo_id = None
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    block_size = 1024
    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if hasattr(config, "max_position_embeddings") and block_size > config.max_position_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length})."
                f"Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that "
                f"default value by passing --block_size xxx."
            )
            block_size = min(1024, config.max_position_embeddings)
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.preprocessed_dataset_path is not None and os.path.exists(args.preprocessed_dataset_path):
        lm_datasets = datasets.load_from_disk(args.preprocessed_dataset_path)
        logger.info(f"Loaded preprocessed dataset from {args.preprocessed_dataset_path}")
    else:
        if args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    split=f"train[:{args.validation_split_percentage}%]",
                )
                raw_datasets["train"] = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    split=f"train[{args.validation_split_percentage}%:]",
                )
        else:
            data_files = {}
            dataset_args = {}
            extension = None
            if args.train_file is not None:
                data_files["train"] = args.train_file
                extension = args.train_file.split(".")[-1]
            if args.validation_file is not None:
                data_files["validation"] = args.validation_file
                extension = args.validation_file.split(".")[-1]
            if extension == "txt":
                extension = "text"
                dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
            raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
            # If no validation data is there, validation_split_percentage will be used to divide the dataset.
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[:{args.validation_split_percentage}%]",
                    **dataset_args,
                )
                raw_datasets["train"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{args.validation_split_percentage}%:]",
                    **dataset_args,
                )

        # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame,
        # etc) at https://huggingface.co/docs/datasets/loading_datasets.

        # Preprocessing the datasets.
        # First we tokenize all the texts.
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        if args.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if args.pad_to_max_length else False

            def tokenize_function(examples):
                # Remove empty lines
                examples[text_column_name] = [
                    line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
                ]
                return tokenizer(
                    examples[text_column_name],
                    padding=padding,
                    truncation=True,
                    max_length=block_size,
                    # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                    # receives the `special_tokens_mask`.
                    return_special_tokens_mask=True,
                )

            with accelerator.main_process_first():
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=args.preprocessing_num_workers,
                    remove_columns=[text_column_name],
                    load_from_cache_file=not args.overwrite_cache,
                    desc="Running tokenizer on dataset line_by_line",
                )
        else:
            # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
            # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
            # efficient when it receives the `special_tokens_mask`.
            def tokenize_function(examples):
                return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

            with accelerator.main_process_first():
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not args.overwrite_cache,
                    desc="Running tokenizer on every text in dataset",
                )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an
            # empty dict. We could add padding if the model supported it instead of this drop, you can customize this
            # part to your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            if args.language_modeling_objective == "clm":
                result["labels"] = result["input_ids"].copy()
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be
        # slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/process#map

        with accelerator.main_process_first():
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
            if args.save_preprocessed_dataset_path is not None:
                lm_datasets.save_to_disk(args.save_preprocessed_dataset_path)
                logger.info(f"Saved preprocessed dataset to {args.save_preprocessed_dataset_path}")
            elif args.preprocessed_dataset_path is not None and not os.path.exists(args.preprocessed_dataset_path):
                lm_datasets.save_to_disk(args.preprocessed_dataset_path)
                logger.info(f"Saved preprocessed dataset to {args.preprocessed_dataset_path}")

    with accelerator.main_process_first():
        lm_datasets.set_format("torch")
        if args.max_train_samples is not None:
            max_train_samples = min(len(lm_datasets["train"]), args.max_train_samples)
            lm_datasets["train"] = lm_datasets["train"].select(range(max_train_samples))
        if args.max_eval_samples is not None:
            max_eval_samples = min(len(lm_datasets["validation"]), args.max_eval_samples)
            lm_datasets["validation"] = lm_datasets["validation"].select(range(max_eval_samples))
        train_dataset = lm_datasets["train"]
        eval_dataset = lm_datasets["validation"]

    # Conditional for small test subsets
    if len(train_dataset) > 3:
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Data collator
    if args.language_modeling_objective == "clm":
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    else:
        # This one will take care of randomly masking the tokens.
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    # we shuffle to get a new random sample each time we evaluate for eval_iters
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    shuffled_eval_dataloader = DataLoader(
        eval_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size,
        generator=generator
    )
    # for evaluation at the end of training
    eval_dataloader = DataLoader(
        eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    model_kwargs = {
        "config": config,
        "low_cpu_mem_usage": args.low_cpu_mem_usage,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.model_name_or_path:
        model_kwargs["pretrained_model_name_or_path"] = args.model_name_or_path
        model_kwargs["from_tf"] = bool(".ckpt" in args.model_name_or_path)
    if args.torch_dtype != "auto":
        model_kwargs["torch_dtype"] = args.torch_dtype
    if args.use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    if args.language_modeling_objective == "clm":
        if args.model_name_or_path:
            model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        else:
            logger.info("Training new model from scratch")
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)
    else:
        if args.model_name_or_path:
            model = AutoModelForMaskedLM.from_pretrained(**model_kwargs)
        else:
            logger.info("Training new model from scratch")
            model = AutoModelForMaskedLM.from_config(config, trust_remote_code=args.trust_remote_code)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized token embeddings from {embedding_size} to {len(tokenizer)}")

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon,
        betas=(args.beta1, args.beta2))

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, shuffled_eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, shuffled_eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    eval_steps = args.eval_steps
    if eval_steps is not None and eval_steps.isdigit():
        eval_steps = int(eval_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initialize automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        if args.run_name is None and args.output_dir is not None:
            logger.info("No run name provided, using the output directory name.")
            args.run_name = f"{args.output_dir.split('/')[-1]}"
        accelerator.init_trackers(project_name=args.project_name, config=experiment_config,
                                  init_kwargs={"wandb": {"name": args.run_name}} if args.run_name else {})

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    resume_step = None
    perplexity = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        batch_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    train_step_loss = loss.detach().float()
                    total_loss += train_step_loss
                    batch_loss += train_step_loss / accelerator.gradient_accumulation_steps
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.with_tracking and accelerator.is_main_process:
                    # https://github.com/huggingface/accelerate/issues/639 loss tracking with gradient accumulation
                    # script seems to get stuck when calling accelerator.gather on batch loss, so we only track
                    # average train loss on the main process
                    train_loss = float(batch_loss)
                    train_perplexity = math.exp(train_loss)
                    log_dict = {
                        "train/loss": train_loss,
                        "train/perplexity": train_perplexity,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "consumed_train_tokens": completed_steps * total_batch_size * block_size
                    }
                    progress_bar.set_postfix(log_dict)
                    accelerator.log(log_dict, step=completed_steps)
                    batch_loss = 0

                if isinstance(eval_steps, int):
                    if completed_steps % eval_steps == 0:
                        logger.info(f"epoch {epoch}: step {completed_steps}: evaluating model")
                        eval_loss, perplexity = validate_model(
                            model, shuffled_eval_dataloader, accelerator, args.per_device_eval_batch_size, args.eval_iters
                        )
                        logger.info(
                            f"epoch {epoch}: step {completed_steps}: perplexity: {perplexity} eval_loss: {eval_loss}")
                        if args.with_tracking:
                            accelerator.log({
                                "eval/loss": eval_loss,
                                "eval/perplexity": perplexity,
                                "epoch": epoch,
                                "step": completed_steps,
                                "consumed_train_tokens": completed_steps * total_batch_size * block_size
                            }, step=completed_steps)
                        model.train()

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)
                if completed_steps >= args.max_train_steps:
                    break

        if args.eval_steps == "epoch":
            # Evaluate on the whole validation split at the end of each epoch
            logger.info(f"epoch {epoch}: step {completed_steps}: evaluating model")
            eval_loss, perplexity = validate_model(
                model, eval_dataloader, accelerator, args.per_device_eval_batch_size
            )
            logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

            if args.with_tracking:
                accelerator.log(
                    {
                        "eval/perplexity": perplexity,
                        "eval/loss": eval_loss,
                        "train/epoch_loss": total_loss.item() / len(train_dataloader),
                        "epoch": epoch,
                        "step": completed_steps,
                        "consumed_train_tokens": completed_steps * total_batch_size * block_size
                    },
                    step=completed_steps,
                )

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                api.upload_folder(
                    commit_message=f"Training in progress epoch {epoch}",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.eval_steps != "epoch":
        # Evaluate on the whole validation split at the end of training
        logger.info(f"End of training: step {completed_steps}: evaluating model")
        eval_loss, perplexity = validate_model(
            model, eval_dataloader, accelerator, args.per_device_eval_batch_size
        )
        logger.info(f"End of training: perplexity: {perplexity} loss: {eval_loss}")
        if args.with_tracking:
            consumed_train_tokens = completed_steps * total_batch_size * block_size
            accelerator.log(
                {
                    "test/perplexity": perplexity,
                    "test/loss": eval_loss,
                    "test/step": completed_steps,
                    "test/consumed_train_tokens": consumed_train_tokens
                },
                step=completed_steps,
            )

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                f.write(json.dumps({"perplexity": perplexity}))


if __name__ == "__main__":
    main()
