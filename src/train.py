#!/usr/bin/env python
# coding=utf-8


""" Finetuning a Transformers model on Super_Glue and Glue tasks."""
# importing the required libraries
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from itertools import chain
import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# to clean up the gpu memory
torch.cuda.empty_cache()

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

# change these paths to your own paths
os.environ["TRANSFORMERS_CACHE"] = "/blue/eel6825/<username>/.cache"
os.environ["HF_HOME"] = "/blue/eel6825/<username>/.cache"
os.environ["XDG_CACHE_HOME"] = "/blue/eel6825/<username>/.cache"


logger = get_logger(__name__)

# Supported tasks are from glue, and super glue, we map the task names to the corresponding dataset columns.
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "boolq": ("passage", "question"),
    "rte": ("premise", "hypothesis"),
    "cb": ("premise", "hypothesis"),
    "record": ("passage", "query", "entities", "answers"),
    "multirc": ("paragraph", "question", "answer"),
    "wic": ("sentence1", "sentence2", "word"),
    "wsc": ("text", "span1_text", "span2_text"),
    "copa": ("premise", "choice1", "choice2", "question"),
}


def parse_args():
    """ Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on SuperGLUE/GLUE tasks."
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the super_glue or glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="super_glue",
        help="The name of the dataset (super_glue & glue) to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
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
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--clip", type=float, default=0.0, help="Gradient Clipping to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
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
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
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
        "--classifier_dropout",
        type=float,
        default=0.0,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    args = parser.parse_args()

    # Sanity checks
    if (
        args.task_name is None
        and args.train_file is None
        and args.validation_file is None
    ):
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
            ], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
            ], "`validation_file` should be a csv or a json file."

    return args


def main():
    args = parse_args()
    # Setup Accelerator to use the appropriate distributed backend
    accelerator = Accelerator("cuda")
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # log the accelerator state (scale_factor, local_rank, etc.)
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

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a Super GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (
            args.train_file if args.train_file is not None else args.validation_file
        ).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

    # Labels are counted for each task during preprocessing
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            if args.task_name not in ["record", "wsc", "wic", "multirc", "copa"]:
                label_list = raw_datasets["train"].features["label"].names
            else:
                label_list = [0, 1]
            num_labels = len(label_list)
        else:
            num_labels = 1

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name
    )
    config.classifier_dropout = args.classifier_dropout
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer
    )  # get the tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )  # get the model

    # Preprocessing the datasets based on the task
    if args.task_name is not None:
        if args.task_name not in ["record", "wsc", "wic", "multirc", "copa"]:
            sentence1_key, sentence2_key = task_to_keys[args.task_name]
        elif args.task_name == "record":
            passage_key, query_key, entities_key, label_column_name = task_to_keys[
                args.task_name
            ]
        elif args.task_name == "multirc":
            paragraph_key, question_key, answers_key = task_to_keys[args.task_name]
        elif args.task_name == "wic":
            sentence1_key, sentence2_key, word_key = task_to_keys[args.task_name]
        elif args.task_name == "wsc":
            text_key, span1_text_key, span2_text_key = task_to_keys[args.task_name]
        elif args.task_name == "copa":
            premise_key, choice1_key, choice2_key, question_key = task_to_keys[
                args.task_name
            ]

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {
                i: label_name_to_id[label_list[i]] for i in range(num_labels)
            }
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False
    if ("gpt" in args.model_name_or_path) or (
        "transfo-xl" in args.model_name_or_path
    ):  # add padding and separation tokens for gpt and transformer-xl models.
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        tokenizer.sep_token = tokenizer.eos_token
        model.config.sep_token_id = model.config.eos_token_id

    def preprocess_function(examples):
        """Preprocess the examples for the task.

        Args:
            examples (dict): The examples to be preprocessed

        Returns:
            dict: The preprocessed examples
        """
        if args.task_name not in [
            "record",
            "wsc",
            "wic",
            "multirc",
            "copa",
        ]:  # Handle most of the GLUE tasks as well as SuperGLUE which have 2 sentences as input
            # Group the 2 sentences together and chunk them into chunks of max allowed length using approriate tokenizer
            texts = (
                (examples[sentence1_key],)
                if sentence2_key is None
                else (examples[sentence1_key], examples[sentence2_key])
            )

            # Tokenize the texts
            result = tokenizer(
                *texts, padding=padding, max_length=args.max_length, truncation=True
            )
            # Store the labels
            result["labels"] = examples["label"]
        elif args.task_name == "record":  # handle record task
            first_sentences = []
            # Duplicate contexts for each possible solution
            for i, context in enumerate(examples[passage_key]):
                options = len(examples[entities_key][i])
                context_dups = [context] * options
                first_sentences.append(context_dups)

            question_headers = examples[query_key]
            second_sentences = []
            labels = []
            # Replace @placeholder with each possible solution and concate with question
            for i, header in enumerate(question_headers):
                possible_solutions = []
                for entity in examples[entities_key][i]:
                    possible_solutions.append(header.replace("@placeholder", entity))
                    answers = examples[label_column_name][i]
                    if answers != []:
                        label = 1 if entity in answers else 0
                    else:
                        label = -1
                    labels.append(label)
                second_sentences.append(possible_solutions)

            # Flatten out the sentences
            first_sentences = list(chain(*first_sentences))
            second_sentences = list(chain(*second_sentences))

            # Tokenize the texts
            tokenized_examples = tokenizer(
                first_sentences,
                second_sentences,
                padding=padding,
                max_length=args.max_length,
                truncation=True,
            )
            # Store the labels
            tokenized_examples["labels"] = labels

            result = tokenized_examples
        elif args.task_name == "multirc":  # handle multirc task
            contexts = []
            # concatenate paragraph and question
            for paragraph, question in zip(
                examples[paragraph_key], examples[question_key]
            ):
                contexts.append(paragraph + " " + question)

            # Tokenize the texts
            result = tokenizer(
                contexts,
                examples[answers_key],
                padding=padding,
                max_length=args.max_length,
                truncation=True,
            )
            # Store the labels
            result["labels"] = examples["label"]
        elif args.task_name == "wic":  # handle wic task
            sentences = []
            # concatenate sentence1 and sentence2 with the word
            for parts in zip(
                *[examples[sentence1_key], examples[sentence2_key], examples[word_key]]
            ):
                sentences.append(tokenizer.sep_token.join(parts))

            # Tokenize the texts
            result = tokenizer(
                sentences, padding=padding, max_length=args.max_length, truncation=True
            )
            # Store the labels
            result["labels"] = examples["label"]
        elif args.task_name == "wsc":  # handle wsc task
            sentences = []
            # concatenate text with the 2 coreference words
            for parts in zip(
                *[
                    examples[text_key],
                    examples[span1_text_key],
                    examples[span2_text_key],
                ]
            ):
                sentences.append(tokenizer.sep_token.join(parts))
            # Tokenize the texts
            result = tokenizer(
                sentences, padding=padding, max_length=args.max_length, truncation=True
            )
            # Store the labels
            result["labels"] = examples["label"]
        elif args.task_name == "copa":  # handle copa task

            choice1_clause = []
            # concatenate context with choice1
            for context, choice in zip(examples[premise_key], examples[choice1_key]):
                choice1_clause.append(context + " " + choice)

            choice2_clause = []
            # concatenate context with choice2
            for context, choice in zip(examples[premise_key], examples[choice2_key]):
                choice2_clause.append(context + " " + choice)

            # Tokenize the texts
            result = tokenizer(
                choice1_clause,
                choice2_clause,
                padding=padding,
                max_length=args.max_length,
                truncation=True,
            )
            # Store the labels
            result["labels"] = examples["label"]

        return result

    with accelerator.main_process_first():
        # Preprocessing the datasets.
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )
    # Get the train and validation datasets
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets[
        "validation_matched" if args.task_name == "mnli" else "validation"
    ]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    # Create the DataLoaders for our training and validation sets.
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Get the metric function based on the type of task
    if args.task_name is not None and args.task_name not in ["record", "multirc"]:
        metric = evaluate.load(args.dataset_name, args.task_name)
    else:
        metric = evaluate.load("accuracy")

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
    best_metric = []
    # Main training loop
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()  # Set the model to training mode (apply dropout etc)
        for step, batch in enumerate(
            train_dataloader
        ):  # Loop over all batches in one epoch
            # We need to skip steps until we reach the resumed step
            if (
                args.resume_from_checkpoint and epoch == starting_epoch
            ):  # If we are resuming training
                if resume_step is not None and step < resume_step:
                    completed_steps += 1  # Skip past steps we have already completed
                    continue
            outputs = model(**batch)  # Forward pass
            loss = outputs.loss  # Get the loss
            loss = (
                loss / args.gradient_accumulation_steps
            )  # Normalize our loss (if averaged)
            accelerator.backward(loss)  # Backward pass
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.clip
            )  # Clip gradients with passed clip parameter
            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):  # Update the model weights
                optimizer.step()  # Update the optimizer
                lr_scheduler.step()  # Update the learning rate scheduler
                optimizer.zero_grad()  # Reset gradients
                progress_bar.update(1)  # Update the progress bar
                completed_steps += 1  # Update the completed steps

            if isinstance(
                checkpointing_steps, int
            ):  # Save the model checkpoint every checkpointing_steps steps
                if (
                    completed_steps % checkpointing_steps == 0
                ):  # If we are on a checkpointing step
                    output_dir = (
                        f"step_{completed_steps }"  # Create the output directory
                    )
                    if args.output_dir is not None:  # If we have an output directory
                        output_dir = os.path.join(
                            args.output_dir, output_dir
                        )  # Set the output directory to the passed output directory
                    accelerator.save_state(output_dir)  # Save the model

            if completed_steps >= args.max_train_steps:  # If we have completed training
                break  # Break from the training loop

        model.eval()  # Set the model to evaluation mode (apply dropout etc)
        samples_seen = 0
        for step, batch in enumerate(
            eval_dataloader
        ):  # Loop over all batches in one epoch
            with torch.no_grad():  # Do not calculate gradients
                outputs = model(**batch)  # Forward pass
            predictions = (
                outputs.logits.argmax(dim=-1)
                if not is_regression
                else outputs.logits.squeeze()
            )  # Get the predicted class
            predictions, references = accelerator.gather(
                (predictions, batch["labels"])
            )  # Gather predictions and references from all processes
            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[
                        : len(eval_dataloader.dataset) - samples_seen
                    ]
                    references = references[
                        : len(eval_dataloader.dataset) - samples_seen
                    ]
                else:
                    samples_seen += references.shape[0]

            metric.add_batch(
                predictions=predictions, references=references,
            )  # Add the batch to our metric
        eval_metric = metric.compute()  # Compute the metric
        logger.info(f"epoch {epoch}: {eval_metric}")  # Log the metric

        if args.checkpointing_steps == "epoch":  # Save the model checkpoint every epoch
            if args.output_dir is not None:  # If we have an output directory
                output_dir = os.path.join(
                    args.output_dir
                )  # Set the output directory to the passed output directory
            if best_metric == []:
                # accelerator.save_state(output_dir)
                model.save_pretrained(
                    args.output_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                )  # Save the model
                best_metric = eval_metric  # Update the best metric
                print("Saved Best Model!!")
            else:
                best_metrics = list(best_metric.values())  # Get the best metric
                curr_metrics = list(eval_metric.values())  # Get the current metric
                save = True
                for i in range(len(curr_metrics)):  # Loop over all metrics
                    if (
                        curr_metrics[i] < best_metrics[i]
                    ):  # If the current metric is worse than the best metric
                        save = False  # Do not save the model
                if save:  # If we should save the model
                    model.save_pretrained(
                        args.output_dir,
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                    )  # Save the model
                    best_metric = eval_metric  # Update the best metric
                    print("Saved Best Model!!")

    if args.output_dir is not None:  # If we have an output directory
        accelerator.wait_for_everyone()  # Wait for all processes to finish before saving the tokenizer
        unwrapped_model = accelerator.unwrap_model(model)  # Unwrap the model
        best_metrics = list(best_metric.values())  # Get the best metric
        curr_metrics = list(eval_metric.values())  # Get the current metric
        save = True
        for i in range(len(curr_metrics)):  # Loop over all metrics
            if (
                curr_metrics[i] < best_metrics[i]
            ):  # If the current metric is worse than the best metric
                save = False  # Do not save the model
        if save:  # If we should save the model
            unwrapped_model.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )  # Save the model
        if accelerator.is_main_process:  # If we are the main process
            tokenizer.save_pretrained(args.output_dir)  # Save the tokenizer

    if args.task_name == "mnli":  # If we are training MNLI task from GLUE
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets[
            "validation_mismatched"
        ]  # Get the validation mismatched dataset
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=data_collator,
            batch_size=args.per_device_eval_batch_size,
        )  # Create the evaluation dataloader
        eval_dataloader = accelerator.prepare(
            eval_dataloader
        )  # Prepare the evaluation dataloader

        model.eval()  # Set the model to evaluation mode (apply dropout etc)
        for step, batch in enumerate(
            eval_dataloader
        ):  # Loop over all batches in one epoch
            outputs = model(**batch)  # Forward pass
            predictions = outputs.logits.argmax(dim=-1)  # Get the predicted class
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )  # Add the batch to our metric

        eval_metric = metric.compute()  # Compute the metric
        logger.info(f"mnli-mm: {eval_metric}")  # Log the metric

    if args.output_dir is not None:  # If we have an output directory
        all_results = {
            f"eval_{k}": v for k, v in best_metric.items()
        }  # Get the best metric
        with open(
            os.path.join(args.output_dir, "all_results.json"), "w"
        ) as f:  # Open the file
            json.dump(all_results, f)  # Save the best metric to a file


if __name__ == "__main__":
    main()  # Run the main function
