"""
Fine-tuned CodeT5 for Python Code Generation
Task: Natural Language Docstring → Python Code
Model: Salesforce/codet5-base
Dataset: CodeSearchNet Python
"""

import os
import json
import logging
import numpy as np

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import evaluate as hf_evaluate
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME = "Salesforce/codet5-base"
OUTPUT_DIR = "./outputs/codet5-python-codegen"
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
NUM_TRAIN_EPOCHS = 5
LEARNING_RATE = 5e-5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIMIT = 2
SEED = 42

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class CodeGenDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_input_len, max_target_len):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        docstring = item.get("func_documentation_string", "") or item.get("docstring", "")
        code = item.get("func_code_string", "") or item.get("code", "")

        source = f"Generate Python: {docstring.strip()}"
        target = code.strip()

        model_inputs = self.tokenizer(
            source,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            target,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = model_inputs["input_ids"].squeeze()
        attention_mask = model_inputs["attention_mask"].squeeze()
        label_ids = labels["input_ids"].squeeze()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
        }


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

def load_data():
    logger.info("📦 Loading CodeSearchNet Python dataset...")
    dataset = load_dataset("code_search_net", "python", trust_remote_code=True)
    logger.info(f"  Train size : {len(dataset['train'])}")
    logger.info(f"  Valid size : {len(dataset['validation'])}")
    logger.info(f"  Test  size : {len(dataset['test'])}")
    return dataset


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def build_compute_metrics(tokenizer):
    bleu_metric = hf_evaluate.load("sacrebleu")
    rouge_metric = hf_evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = np.clip(preds, 0, tokenizer.vocab_size - 1)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        bleu_result = bleu_metric.compute(
            predictions=decoded_preds,
            references=[[l] for l in decoded_labels],
        )
        rouge_result = rouge_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
        )

        return {
            "bleu": round(bleu_result["score"], 4),
            "rouge1": round(rouge_result["rouge1"], 4),
            "rougeL": round(rouge_result["rougeL"], 4),
        }

    return compute_metrics


# ─────────────────────────────────────────────
# MAIN TRAINING LOOP
# ─────────────────────────────────────────────

def train():
    torch.manual_seed(SEED)

    logger.info(f"🤗 Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total parameters: {total_params:,}")

    raw_dataset = load_data()

    # Uncomment untuk test cepat dulu:
    # raw_dataset["train"] = raw_dataset["train"].select(range(5000))
    # raw_dataset["validation"] = raw_dataset["validation"].select(range(500))

    train_dataset = CodeGenDataset(raw_dataset["train"], tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    val_dataset   = CodeGenDataset(raw_dataset["validation"], tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        fp16=torch.cuda.is_available(),
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to="none",
        seed=SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    logger.info("🚀 Starting training...")
    train_result = trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("✅ Training complete!")
    logger.info(f"   Model saved to: {OUTPUT_DIR}")

    return trainer, tokenizer


if __name__ == "__main__":
    trainer, tokenizer = train()
    raw_dataset = load_data()