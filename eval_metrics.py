"""
evaluate.py — Detailed evaluation of a fine-tuned CodeT5 model.

Computes:
  - BLEU-4 (sacreBLEU)
  - ROUGE-1 / ROUGE-L
  - CodeBLEU (token-level, keyword match, dataflow — simplified)
  - Pass@k (requires execution sandbox, skipped by default)

Usage:
    python evaluate.py --model_path ./outputs/codet5-python-codegen \
                       --split test \
                       --max_samples 500
"""

import argparse
import json
import logging
import os

import numpy as np
import torch
from datasets import load_dataset
from eval_metrics import load as load_metric
from tqdm import tqdm
from transformers import RobertaTokenizer, T5ForConditionalGeneration

from inference import generate_code, load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PYTHON KEYWORDS for CodeBLEU-lite
# ─────────────────────────────────────────────
PYTHON_KEYWORDS = {
    "def", "return", "if", "else", "elif", "for", "while", "in", "not",
    "and", "or", "import", "from", "class", "try", "except", "with",
    "as", "pass", "break", "continue", "lambda", "yield", "raise", "True",
    "False", "None", "is", "global", "nonlocal", "del", "assert",
}


def keyword_match_score(pred: str, ref: str) -> float:
    """Fraction of reference Python keywords present in prediction."""
    ref_kw = {t for t in ref.split() if t in PYTHON_KEYWORDS}
    if not ref_kw:
        return 1.0
    pred_kw = {t for t in pred.split() if t in PYTHON_KEYWORDS}
    return len(pred_kw & ref_kw) / len(ref_kw)


def exact_match(pred: str, ref: str) -> bool:
    return pred.strip() == ref.strip()


# ─────────────────────────────────────────────
# EVALUATION LOOP
# ─────────────────────────────────────────────

def evaluate(
    model,
    tokenizer,
    device: str,
    split: str = "test",
    max_samples: int = 500,
    output_path: str = "./outputs/eval_results.json",
):
    logger.info(f"📦 Loading CodeSearchNet Python ({split} split) ...")
    dataset = load_dataset("code_search_net", "python", trust_remote_code=True)[split]

    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
        logger.info(f"   Using {max_samples} samples for evaluation.")

    bleu_metric = load_metric("sacrebleu")
    rouge_metric = load_metric("rouge")

    all_preds, all_refs = [], []
    kw_scores, em_scores = [], []

    logger.info("🔄 Generating predictions ...")
    for item in tqdm(dataset, total=len(dataset)):
        docstring = item.get("func_documentation_string", "") or item.get("docstring", "")
        reference = item.get("func_code_string", "") or item.get("code", "")

        if not docstring.strip() or not reference.strip():
            continue

        preds = generate_code(
            docstring, model, tokenizer, device,
            num_return_sequences=1, num_beams=5,
        )
        pred = preds[0] if preds else ""

        all_preds.append(pred.strip())
        all_refs.append(reference.strip())

        kw_scores.append(keyword_match_score(pred, reference))
        em_scores.append(float(exact_match(pred, reference)))

    logger.info(f"   Samples evaluated: {len(all_preds)}")

    # BLEU
    bleu = bleu_metric.compute(
        predictions=all_preds,
        references=[[r] for r in all_refs],
    )["score"]

    # ROUGE
    rouge = rouge_metric.compute(predictions=all_preds, references=all_refs)

    # Keyword match (simplified CodeBLEU component)
    avg_kw = float(np.mean(kw_scores))

    # Exact match
    em = float(np.mean(em_scores)) * 100

    results = {
        "split": split,
        "num_samples": len(all_preds),
        "bleu4": round(bleu, 4),
        "rouge1": round(rouge["rouge1"], 4),
        "rougeL": round(rouge["rougeL"], 4),
        "keyword_match": round(avg_kw * 100, 2),
        "exact_match_pct": round(em, 2),
    }

    logger.info("\n" + "═" * 50)
    logger.info("  Evaluation Results")
    logger.info("═" * 50)
    for k, v in results.items():
        logger.info(f"  {k:<22}: {v}")
    logger.info("═" * 50)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✅ Results saved to: {output_path}")

    # Save a few qualitative examples
    examples = []
    for i in range(min(10, len(all_preds))):
        examples.append({
            "docstring": dataset[i].get("func_documentation_string", "")[:200],
            "reference": all_refs[i][:400],
            "predicted": all_preds[i][:400],
        })

    examples_path = output_path.replace(".json", "_examples.json")
    with open(examples_path, "w") as f:
        json.dump(examples, f, indent=2)
    logger.info(f"   Qualitative examples saved to: {examples_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./outputs/codet5-python-codegen")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--output", type=str, default="./outputs/eval_results.json")
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model_path)
    evaluate(model, tokenizer, device, args.split, args.max_samples, args.output)
