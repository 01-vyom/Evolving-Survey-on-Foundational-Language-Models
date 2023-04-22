
import argparse
import json
import logging
import os
import random
from itertools import chain
import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
torch.cuda.empty_cache()

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
    T5ForConditionalGeneration
)

os.environ['TRANSFORMERS_CACHE'] = '/blue/eel6825/v.pathak/.cache'
os.environ['HF_HOME'] = '/blue/eel6825/v.pathak/.cache'
os.environ['XDG_CACHE_HOME'] = '/blue/eel6825/v.pathak/.cache'

logger = get_logger(__name__)

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
    "wsc": ("text", "span1_text", "span2_text", "span1_index", "span2_index"),
    "copa": ("premise", "choice1", "choice2", "question")
}
COPA_DICT = {
        "cause": "What was the cause of this?",
        "effect": "What happened as a result of this?",
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a transformers model on SuperGLUE")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the super glue task to train on.",
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
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
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
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible evaluation.")
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    return args

def main():
    args = parse_args()
    
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = Accelerator("cuda") # A100 needed
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

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
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
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    
    # Labels
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
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    if ("t5" in args.model_name_or_path):
        model = T5ForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )
        
    # Preprocessing the datasets
    if args.task_name is not None:
        if args.task_name not in ["record", "wsc", "wic", "multirc", "copa"]:
            sentence1_key, sentence2_key = task_to_keys[args.task_name]
        elif args.task_name == "record":
            passage_key, query_key, entities_key, label_column_name = task_to_keys[args.task_name]
        elif args.task_name == "multirc":
            paragraph_key, question_key, answers_key = task_to_keys[args.task_name]
        elif args.task_name == "wic":
            sentence1_key, sentence2_key, word_key = task_to_keys[args.task_name]
        elif args.task_name == "wsc":
            text_key, span1_text_key, span2_text_key, span1_idx_key, span2_idx_key = task_to_keys[args.task_name]
        elif args.task_name == "copa":
            premise_key, choice1_key, choice2_key, question_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None


    padding = "max_length" if args.pad_to_max_length else False
    if "t5" in args.model_name_or_path: #heuristic
        padding = True
    
    def preprocess_function(examples):
        if "t5" in args.model_name_or_path:
            sentences = []
            if args.task_name == "boolq":
                for question, context in zip(examples[sentence1_key], examples[sentence2_key]):
                    input_text = f"boolq question: {question} context: {context}"
                    sentences.append(input_text)
            elif args.task_name == "rte":
                for sentence1, sentence2 in zip(examples[sentence1_key], examples[sentence2_key]):
                    input_text = f"rte sentence1: {sentence1} sentence2: {sentence2}"
                    sentences.append(input_text)
            elif args.task_name == "cb":
                for premise, hypothesis in zip(examples[sentence1_key], examples[sentence2_key]):
                    input_text = f"cb hypothesis: {hypothesis} premise: {premise}"
                    sentences.append(input_text)
            elif args.task_name == "copa":
                for premise, choice1, choice2, question in zip(examples[premise_key], examples[choice1_key], examples[choice2_key], examples[question_key]):
                    input_text = f"copa choice1: {choice1} choice2: {choice2} premise: {premise} question: {COPA_DICT[question]}"
                    sentences.append(input_text)
            elif args.task_name == "wsc":
                for premise, span2_idx, span1_idx  in zip(examples[text_key], examples[span2_idx_key], examples[span1_idx_key]):
                    premises = premise.split(" ")
                    premises[span2_idx] = "*"+premises[span2_idx]+"*"
                    premise = " ".join(premises)
                    input_text = f"wsc sentence: {premise}"
                    sentences.append(input_text)
            elif args.task_name == "wic":
                for sentence1, sentence2, word in zip(examples[sentence1_key], examples[sentence2_key], examples[word_key]):
                    input_text = f"wic sentence1: {sentence1} sentence2: {sentence2} word: {word}"
                    sentences.append(input_text)
            result = tokenizer(
                    sentences,
                    padding=padding, 
                    max_length=args.max_length, 
                    truncation=True)
            result["labels"] = examples["label"]
            return result
        else:
            if args.task_name not in ["record", "wsc", "wic", "multirc", "copa"]:
                # Tokenize the texts
                texts = (
                    (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
                )
                result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
            elif args.task_name == "record":
                first_sentences = []
                for i, context in enumerate(examples[passage_key]):
                    options = len(examples[entities_key][i])
                    context_dups = [context] * options
                    first_sentences.append(context_dups)
                    
                question_headers = examples[query_key]
                second_sentences = []
                labels = []
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
                
                

                # Flatten out
                first_sentences = list(chain(*first_sentences))
                second_sentences = list(chain(*second_sentences))
                
                
                tokenized_examples = tokenizer(
                                            first_sentences,
                                            second_sentences,
                                            padding=padding, 
                                            max_length=args.max_length, 
                                            truncation=True)
                
                tokenized_examples['labels'] = labels
                
                result = tokenized_examples
            elif args.task_name == "multirc":
                contexts = []
                for paragraph, question in zip(examples[paragraph_key], examples[question_key]):
                    contexts.append(paragraph + " " + question)
                
                result = tokenizer(
                                    contexts,
                                examples[answers_key], 
                                padding=padding, 
                                max_length=args.max_length, 
                                truncation=True)
                result["labels"] = examples["label"]
            elif args.task_name == "wic":
                sentences = []
                for parts in zip(*[examples[sentence1_key], examples[sentence2_key], examples[word_key]]):
                    sentences.append(tokenizer.sep_token.join(parts))
                result = tokenizer(
                                    sentences,
                                    padding=padding, 
                                    max_length=args.max_length, 
                                    truncation=True)
                result["labels"] = examples["label"]
            elif args.task_name == "wsc":
                sentences = []
                for parts in zip(*[examples[text_key], examples[span1_text_key], examples[span2_text_key]]):
                    sentences.append(tokenizer.sep_token.join(parts))
                result = tokenizer(
                                    sentences,
                                    padding=padding, 
                                    max_length=args.max_length, 
                                    truncation=True)
                result["labels"] = examples["label"]
            elif args.task_name == "copa":
                # contexts = []
                # for premise, question in zip(examples[premise_key], examples[question_key]):
                #     contexts.append(premise + " " + COPA_DICT[question])
                
                choice1_clause = []
                for context, choice in zip(examples[premise_key], examples[choice1_key]):
                    choice1_clause.append(context + " " + choice)
                
                choice2_clause = []
                for context, choice in zip(examples[premise_key], examples[choice2_key]):
                    choice2_clause.append(context + " " + choice)
                
                result = tokenizer(
                                    choice1_clause,
                                    choice2_clause,
                                    padding=padding, 
                                    max_length=args.max_length, 
                                    truncation=True)
                result["labels"] = examples["label"]
                    
            return result
    
    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )
    
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]
    
    # Log a few random samples from the training set:
    for index in random.sample(range(len(eval_dataset)), 3):
        logger.info(f"Sample {index} of the validation set: {eval_dataset[index]}.")
    
    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))
    
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    
    # Prepare everything with our `accelerator`.
    model, eval_dataloader, = accelerator.prepare(
        model, eval_dataloader
    )
    
    # Get the metric function
    if args.task_name is not None and args.task_name not in ["record", "multirc"]:
        metric = evaluate.load(args.dataset_name, args.task_name)
    else:
        metric = evaluate.load("accuracy")
        
    logger.info("***** Running inference *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
    
    model.eval()
    samples_seen = 0
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            if "t5" in args.model_name_or_path:
                outputs = model.generate(**batch)
            else:
                outputs = model(**batch)
        if "t5" in args.model_name_or_path:
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            format_predictions = []
            print(predictions)
            for prediction in predictions:
                if args.task_name == "copa":
                    format_predictions.append(prediction.lower() == "choice2")
                elif args.task_name == "boolq":
                    format_predictions.append(prediction.lower() == "true")
                elif args.task_name == "rte":
                    format_predictions.append(prediction.lower() != "entailment")
                elif args.task_name == "cb":
                    format_predictions.append(0 if prediction.lower() == "entailment" else 1 if prediction.lower() == "contradiction" else 2)
                elif args.task_name == "wsc": 
                    format_predictions.append(prediction.lower() != "acceptable")
                elif args.task_name == "wic":
                    format_predictions.append(prediction.lower() == "true")
            print(format_predictions, batch["labels"])
            print("---")
            predictions = format_predictions
        else:
            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
        predictions, references = accelerator.gather((predictions, batch["labels"]))
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(eval_dataloader) - 1:
                predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                references = references[: len(eval_dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]
                
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
    eval_metric = metric.compute()
    logger.info(f"Eval Results: {eval_metric}")

    if args.output_dir is not None:
        best_metric = eval_metric
        all_results = {f"eval_{k}": v for k, v in best_metric.items()}
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(all_results, f)         

if __name__ == "__main__":
    main()
