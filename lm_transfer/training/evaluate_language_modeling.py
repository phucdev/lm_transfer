import argparse
import logging
import math
import os
import json
import datasets
import random
import transformers
import torch

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from itertools import chain
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from tqdm.auto import tqdm

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a transformers model on a language modeling task")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Pretrained tokenizer name or path if not the same as model_name.")
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
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
        "--train_file", type=str, default=None, help="A csv, txt or a json file containing the training data."
    )
    parser.add_argument("--validation_file", type=str, required=True, help="A file containing the validation data.")
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker eval, truncate the number of evaluation examples to this value if set.",
    )
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                        help="Batch size for the evaluation dataloader.")
    parser.add_argument("--block_size", type=int, default=512,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
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
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the evaluation results.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible evaluation.")
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
    parser.add_argument("--language_modeling_objective", type=str, choices=["clm", "mlm"], default="mlm",
                        help="Language modeling objective for the model.")
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
    return parser.parse_args()


def validate_model(model, eval_dataloader, accelerator, per_device_eval_batch_size):
    """Evaluates the model on eval_dataloader and returns the evaluation loss and perplexity.
    Args:
        model (:obj:`torch.nn.Module`): The model to evaluate.
        eval_dataloader (:obj:`torch.utils.data.DataLoader`): The evaluation dataloader.
        accelerator (:obj:`accelerate.Accelerator`): The distributed training backend.
        per_device_eval_batch_size (:obj:`int`): The batch size per device.
    Returns:
        :obj:`tuple(torch.FloatTensor, float)`: A tuple with the evaluation loss and the perplexity.
    """
    model.eval()
    losses = []
    num_eval_steps = len(eval_dataloader)
    eval_progress_bar = tqdm(range(num_eval_steps), disable=not accelerator.is_local_main_process,
                             position=0, leave=True, desc="Evaluating")
    for step, batch in enumerate(eval_dataloader):
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

    accelerator = Accelerator()

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

    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"Set seed to {args.seed}")

    # Load model and tokenizer
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

    block_size = min(args.block_size, tokenizer.model_max_length)

    # Load and preprocess dataset
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
        column_names = raw_datasets["validation"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

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
        if args.max_eval_samples is not None:
            max_eval_samples = min(len(lm_datasets["validation"]), args.max_eval_samples)
            lm_datasets["validation"] = lm_datasets["validation"].select(range(max_eval_samples))
        eval_dataset = lm_datasets["validation"]

    # Conditional for small test subsets
    if len(eval_dataset) > 3:
        # Log a few random samples from the training set:
        for index in random.sample(range(len(eval_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {eval_dataset[index]}.")

    # Data collator
    if args.language_modeling_objective == "clm":
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    else:
        # This one will take care of randomly masking the tokens.
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)
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

    # Prepare everything with our `accelerator`.
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    # Evaluate the model
    eval_loss, perplexity = validate_model(model, eval_dataloader, accelerator, args.per_device_eval_batch_size)

    logger.info(f"Evaluation Loss: {eval_loss:.4f}")
    logger.info(f"Evaluation Perplexity: {perplexity:.4f}")

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
            f.write(json.dumps({"perplexity": perplexity}))


if __name__ == "__main__":
    main()
