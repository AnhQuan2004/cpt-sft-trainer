import yaml
import re
import json
from loguru import logger
import datasets
from datasets import load_dataset, concatenate_datasets

def load_yaml_config(path: str) -> dict:
    """Load a yaml file and return a dictionary."""

    try: 
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading yaml file {path}: {e}")
        raise e

def formatting_prompts_func(examples, tokenizer):
    return {"text" : [example + tokenizer.eos_token for example in examples["text"]]}


def load_and_merge_datasets(config: dict) -> datasets.Dataset:
    """Load and merge datasets."""

    datasets = []
    for dataset_name in config["datasets"]["names"]:
        datasets.append(load_dataset(dataset_name, split="train"))

    return concatenate_datasets(datasets).shuffle(seed=3047)

def apply_chat_template(example, tokenizer):

    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. Your goal is to provide accurate and safe information to the user."
        },
        {
            "role": "user",
            "content": str(example['instruction']) + ('\n' + example["input"] if "input" in example and example["input"] else "")},
        {
            "role": "assistant",
            "content": example.get("output", "")}
    ]

    chat_format = tokenizer.apply_chat_template(messages, tokenize=False)
    return {
        'text': chat_format
    }

def format_dpo_dataset(example, tokenizer):
    rejected_messages = [
        {
            "role": "user",
            "content": example['question']},
        {
            "role": "assistant",
            "content": example["rejected"]}
    ]

    chosen_messages = [
        {
            "role": "user",
            "content": example['question']},
        {
            "role": "assistant",
            "content": example['chosen']}
    ]
    
    return {
        'rejected': tokenizer.apply_chat_template(rejected_messages, tokenize=False),
        'chosen': tokenizer.apply_chat_template(chosen_messages, tokenize=False)
    }

def format_goemotions_plain(example, tokenizer):
    """Format GoEmotions data for plain text fine-tuning (no chat template)."""
    text = example["text"]
    emotions = []
    emotion_columns = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    for col in emotion_columns:
        if col in example and example[col] == 1:
            emotions.append(col)

    joined_labels = ", ".join(emotions) if emotions else "neutral"
    
    prompt = f"Input: {text}\nOutput: {joined_labels}"
    return {"text": prompt}
