"""
Inference module for CodeT5 Python Code Generation
"""

import os
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FALLBACK_MODEL = "roybeey/codet5-python-codegen"

MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256
NUM_BEAMS = 5
NUM_RETURN_SEQUENCES = 3


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

def load_model():
    print(f"⬇️ Loading model from HuggingFace: {FALLBACK_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
    model = T5ForConditionalGeneration.from_pretrained(FALLBACK_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"✅ Model loaded on: {device}")
    return model, tokenizer, device


# ─────────────────────────────────────────────
# GENERATE CODE
# ─────────────────────────────────────────────

def generate_code(docstring, model, tokenizer, device,
                  max_length=MAX_TARGET_LENGTH,
                  num_beams=NUM_BEAMS,
                  num_return_sequences=NUM_RETURN_SEQUENCES):
    """
    Generate Python code from a natural language docstring.
    """
    source = f"Generate Python: {docstring.strip()}"

    inputs = tokenizer(
        source,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    results = []
    for output in outputs:
        code = tokenizer.decode(output, skip_special_tokens=True)
        results.append(code.strip())

    return results


# ─────────────────────────────────────────────
# MAIN (test)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    model, tokenizer, device = load_model()

    test_docstrings = [
        "Calculate the factorial of n recursively",
        "Check if a number is prime",
        "Sort a list of integers in ascending order",
    ]

    for docstring in test_docstrings:
        print(f"\n📝 Input: {docstring}")
        print("🔧 Generated Code:")
        results = generate_code(docstring, model, tokenizer, device)
        for i, code in enumerate(results, 1):
            print(f"  [{i}] {code}")