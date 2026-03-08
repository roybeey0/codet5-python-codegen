"""
inference.py — Generate Python code from a natural language docstring
"""

import argparse
import os
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

DEFAULT_MODEL = "./outputs/codet5-python-codegen"
FALLBACK_MODEL = "Salesforce/codet5-base"


def load_model(model_path: str = DEFAULT_MODEL):
    abs_path = os.path.abspath(model_path)

    if os.path.isdir(abs_path):
        use_path = abs_path
        print(f"✅ Loaded fine-tuned model from: {use_path}")
    else:
        use_path = FALLBACK_MODEL
        print(f"⚠️  Fine-tuned model not found. Using base model: {FALLBACK_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(use_path)
    model = T5ForConditionalGeneration.from_pretrained(use_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"🖥️  Device: {device}")
    return model, tokenizer, device


def generate_code(
    docstring: str,
    model,
    tokenizer,
    device: str,
    max_input_length: int = 256,
    max_target_length: int = 256,
    num_beams: int = 5,
    num_return_sequences: int = 3,
    temperature: float = 0.8,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
) -> list:
    source = f"Generate Python: {docstring.strip()}"

    inputs = tokenizer(
        source,
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True,
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_target_length,
            num_beams=max(num_beams, num_return_sequences),
            num_return_sequences=num_return_sequences,
            early_stopping=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=3,
        )

    decoded = [
        tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for out in outputs
    ]
    return decoded


DEMO_DOCSTRINGS = [
    "Calculate the factorial of a given non-negative integer n using recursion.",
    "Return the list of all prime numbers up to n using the Sieve of Eratosthenes.",
    "Merge two sorted lists into a single sorted list.",
    "Flatten a nested list of arbitrary depth into a single flat list.",
    "Compute the Fibonacci sequence up to n terms and return as a list.",
    "Check whether a given string is a palindrome, ignoring case and spaces.",
    "Perform binary search on a sorted list and return the index of the target.",
    "Count the frequency of each word in a string and return a dictionary.",
]


def run_demo(model, tokenizer, device):
    print("\n" + "="*60)
    print("  CodeT5 Python Code Generation - Demo")
    print("="*60)
    for i, docstring in enumerate(DEMO_DOCSTRINGS[:3], 1):
        print(f"\n[Example {i}]")
        print(f"Docstring: {docstring}")
        print("-"*50)
        results = generate_code(docstring, model, tokenizer, device, num_return_sequences=1)
        print("Generated Code:")
        print(results[0] if results else "(no output)")
        print("-"*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--docstring", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--num_results", type=int, default=3)
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model_path)

    if args.demo or args.docstring is None:
        run_demo(model, tokenizer, device)
    else:
        results = generate_code(
            args.docstring, model, tokenizer, device,
            num_return_sequences=args.num_results
        )
        print(f"\nDocstring: {args.docstring}\n")
        for i, code in enumerate(results, 1):
            print(f"-- Candidate {i} --")
            print(code)
            print()
