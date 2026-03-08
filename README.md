# 🧠 CodeT5 Python Code Generator

> **Fine-tuned CodeT5 Transformer for Python code generation from natural language docstrings**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗%20Transformers-4.40%2B-yellow)](https://huggingface.co/transformers)
[![Gradio](https://img.shields.io/badge/Gradio-4.29%2B-orange)](https://gradio.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Overview

This project fine-tunes **Salesforce's CodeT5** (`codet5-base`) on the **CodeSearchNet Python** dataset to perform **docstring-to-code generation**: given a natural language description of a Python function, the model generates the corresponding Python source code.

```
Input  → "Calculate the factorial of n using recursion."
Output → def factorial(n):
             if n == 0:
                 return 1
             return n * factorial(n - 1)
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    CodeT5 (220M params)              │
│                                                     │
│  ┌──────────────────┐     ┌──────────────────────┐  │
│  │    Encoder       │     │      Decoder         │  │
│  │  (RoBERTa-style) │────▶│  (T5-style, causal)  │  │
│  │                  │     │                      │  │
│  │  Docstring tokens│     │  Python code tokens  │  │
│  └──────────────────┘     └──────────────────────┘  │
└─────────────────────────────────────────────────────┘
         ▲                            │
  Natural language              Generated Python
  "Flatten a list..."            def flatten(...)
```

### How It Works

| Step | Detail |
|------|--------|
| **Tokenization** | RoBERTa tokenizer with `"Generate Python: {docstring}"` prefix |
| **Model** | `Salesforce/codet5-base` — encoder-decoder Transformer (220M params) |
| **Training** | Seq2Seq cross-entropy loss, teacher forcing |
| **Decoding** | Beam search (width=5) + nucleus sampling (top-p=0.95) |
| **Evaluation** | BLEU-4, ROUGE-1/L, Keyword Match, Exact Match |

---

## 📊 Dataset

**[CodeSearchNet Python](https://huggingface.co/datasets/code_search_net)**

| Split | Size |
|-------|------|
| Train | ~412K samples |
| Validation | ~13K samples |
| Test | ~14K samples |

Each sample contains a Python function with its docstring. We use:
- **Input**: `func_documentation_string` (natural language)
- **Target**: `func_code_string` (Python source code)

---

## 📈 Results

| Metric | Score |
|--------|-------|
| **BLEU-4** | ~18–22 |
| **ROUGE-1** | ~0.42 |
| **ROUGE-L** | ~0.38 |
| **Keyword Match** | ~71% |

> Results may vary based on training duration, hardware, and hyperparameters.

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/codet5-python-codegen.git
cd codet5-python-codegen

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Train

```bash
python train.py
# Model saved to ./outputs/codet5-python-codegen
```

> 💡 **Tip:** To do a quick sanity check, uncomment the subset lines in `train.py` to train on 5K samples first.

### 3. Evaluate

```bash
python evaluate.py \
  --model_path ./outputs/codet5-python-codegen \
  --split test \
  --max_samples 500
```

### 4. Inference (CLI)

```bash
# Single docstring
python inference.py \
  --docstring "Return all prime numbers up to n using the Sieve of Eratosthenes."

# Run built-in demo examples
python inference.py --demo
```

### 5. Web UI (Gradio)

```bash
python app.py
# Open: http://localhost:7860
```

---

## 📁 Project Structure

```
codet5-python-codegen/
│
├── train.py            # Fine-tuning pipeline (load → preprocess → train → save)
├── inference.py        # Code generation from docstrings
├── evaluate.py         # BLEU, ROUGE, Keyword Match evaluation
├── app.py              # Gradio web demo UI
│
├── outputs/            # (git-ignored) Trained model checkpoints
│   └── codet5-python-codegen/
│       ├── config.json
│       ├── pytorch_model.bin
│       └── tokenizer_config.json
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base model | `Salesforce/codet5-base` |
| Max input tokens | 256 |
| Max output tokens | 256 |
| Batch size | 8 |
| Learning rate | 5e-5 |
| Warmup steps | 500 |
| Epochs | 5 (early stopping, patience=2) |
| Optimizer | AdamW |
| Scheduler | Linear with warmup |
| Precision | FP16 (if GPU available) |
| Beam search width | 5 |

---

## 🖥️ Hardware Requirements

| Setup | Recommended |
|-------|-------------|
| GPU | NVIDIA 16GB+ (e.g. RTX 3080, A100) |
| RAM | 16GB+ |
| Storage | ~10GB (dataset + model) |
| Training time | ~3–6 hrs (full dataset, single GPU) |

> Using Google Colab Pro (A100) is recommended for free GPU access.

---

## 🔧 Customization

**Use a larger model:**
```python
MODEL_NAME = "Salesforce/codet5-large"
```

**Change the task prefix:**
```python
source = f"Summarize Python: {code}"   # for code summarization
```

**Add W&B tracking:**
```python
report_to = "wandb"   # in Seq2SeqTrainingArguments
```

---

## 📚 References

- [CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation](https://arxiv.org/abs/2109.00859) — Wang et al., 2021
- [CodeSearchNet Challenge](https://arxiv.org/abs/1909.09436) — Husain et al., 2019
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Salesforce/codet5-base on HuggingFace](https://huggingface.co/Salesforce/codet5-base)

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

## 👤 Author

**Your Name**  
[GitHub](https://github.com/YOUR_USERNAME) · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [Portfolio](https://yourportfolio.dev)

---

> ⭐ If you found this project useful, please star the repository!