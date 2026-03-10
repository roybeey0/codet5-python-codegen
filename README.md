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
| **Evaluation** | BLEU-4, ROUGE-1/L |

---

## 📊 Dataset

**[CodeSearchNet Python](https://huggingface.co/datasets/code_search_net)**

| Split | Full Size | Used |
|-------|------|------|
| Train | ~412K samples | 50K samples |
| Validation | ~23K samples | 2K samples |
| Test | ~22K samples | 22K samples |

Each sample contains a Python function with its docstring. We use:
- **Input**: `func_documentation_string` (natural language)
- **Target**: `func_code_string` (Python source code)

---

## 📈 Training Results

| Metric | Score |
|--------|-------|
| **BLEU-4** | **54.66** |
| **ROUGE-1** | 0.611 |
| **ROUGE-L** | 0.594 |
| **Train Loss** | 1.071 |
| **Epochs** | 3 |
| **Training Time** | ~3h 51m (Kaggle P100) |

---

## ⚠️ Current Limitations

This model was trained on **50K samples (out of 412K)** with only **3 epochs** due to computational constraints. As a result:

- The model works well for **simple, common functions** (file reading, basic data structures)
- The model struggles with **complex algorithms** (recursion, sorting logic, prime checking)
- Generated code may have **structural correctness** but **logical errors**

### To improve model accuracy:

| Improvement | Expected Gain |
|---|---|
| Train on full 412K samples | +10-15% BLEU |
| Increase to 10+ epochs | +5-10% BLEU |
| Use `codet5-large` (770M params) | +15-20% BLEU |
| Use `codet5p-2b` (2B params) | +25-30% BLEU |

> 💡 **Note:** This project is intended as a portfolio demonstration of the fine-tuning pipeline, not as a production-ready code generation tool. For production use, consider training on the full dataset with a larger model.

---

## 🤗 Pre-trained Model

The fine-tuned model is available on HuggingFace Hub:

**[roybeey/codet5-python-codegen](https://huggingface.co/roybeey/codet5-python-codegen)**

```python
from transformers import T5ForConditionalGeneration, AutoTokenizer

model = T5ForConditionalGeneration.from_pretrained("roybeey/codet5-python-codegen")
tokenizer = AutoTokenizer.from_pretrained("roybeey/codet5-python-codegen")
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/roybeey0/codet5-python-codegen.git
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

### 3. Inference (CLI)

```bash
python inference.py
```

### 4. Web UI (Gradio)

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
├── eval_metrics.py     # BLEU, ROUGE evaluation
├── app.py              # Gradio web demo UI
│
├── outputs/            # (git-ignored) Trained model checkpoints
│   └── codet5-python-codegen/
│       ├── config.json
│       ├── model.safetensors
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
| Epochs | 3 |
| Optimizer | AdamW |
| Precision | FP16 (if GPU available) |
| Beam search width | 5 |

---

## 🖥️ Hardware Used

| | Spec |
|-------|-------------|
| GPU | Kaggle P100 (16GB) |
| RAM | 29GB |
| Training time | ~3h 51m |
| Dataset | 50K / 412K samples |

---

## 🔧 Customization

**Use a larger model:**
```python
MODEL_NAME = "Salesforce/codet5-large"
```

**Train on full dataset:**
```python
# Comment out these lines in train.py:
# raw_dataset["train"] = raw_dataset["train"].select(range(50000))
# raw_dataset["validation"] = raw_dataset["validation"].select(range(2000))
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

**roybeey**  
[GitHub](https://github.com/roybeey0) · [HuggingFace](https://huggingface.co/roybeey)

---

> ⭐ If you found this project useful, please star the repository!
