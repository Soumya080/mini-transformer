# Mini-Transformer

A GPT-style language model built from scratch in PyTorch, implementing multi-head causal self-attention, custom feed-forward layers, and full training pipeline without relying on high-level frameworks.

---

## 🚀 Features

* Full Transformer decoder architecture implemented from scratch
* Multi-head causal self-attention with masking
* Custom feed-forward network using `torch.autograd.Function`
* Character-level tokenizer (no external dependencies)
* End-to-end training + text generation pipeline
* Attention visualization support
* Experiment tracking (hyperparameter sweeps)

---

## 📁 Project Structure

```
mini-transformer/
├── src/
│   ├── model/
│   ├── tokenizer/
│   ├── data/
│   └── utils/
├── scripts/
├── tests/
├── experiments/
├── docs/
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

---

## 🧪 Training

```bash
python scripts/train.py
```

* Trains a character-level GPT model on a small corpus
* Hyperparameters are defined inside `train.py`
* Model checkpoint (`model.pt`) is saved after training

---

## 🔍 Attention Visualization

```bash
python scripts/visualize_attention.py
```

* Generates heatmaps of attention weights
* Helps understand token interactions and context flow

---

## 🧠 Architecture Overview

```
Input Tokens
   ↓
Token Embeddings + Positional Embeddings
   ↓
[N × Decoder Blocks]
   ├── Multi-Head Self-Attention (causal mask)
   ├── Residual + LayerNorm
   ├── Feed Forward Network
   └── Residual + LayerNorm
   ↓
Final LayerNorm
   ↓
Linear Projection → Vocabulary logits
```

---

## 🔑 Key Hyperparameters

| Parameter      | Value  |
| -------------- | ------ |
| d_model        | 128    |
| n_heads        | 1      |
| d_ff           | 512    |
| n_layers       | 4      |
| max_seq_len    | 64     |
| batch_size     | 16     |
| learning_rate  | 1e-3   |
| training_steps | 10,000 |

---

## 📊 Experiments

Experiments include hyperparameter sweeps on:

* Number of attention heads
* Model dimension (`d_model`)
* Maximum sequence length
* Number of layers

Results are stored in:

```
experiments/results/
experiments/plots/
```

---

## 📈 Sample Output

*(Add your generated text here after training)*

Example:

```
Input: "The quick brown fox"
Output: "The quick brown fox jumps over the lazy dog..."
```

---

## ⚠️ Limitations

* Character-level modeling limits semantic understanding
* Small dataset leads to limited generalization
* No optimization techniques (e.g., learning rate scheduling, dropout tuning)
* No batching optimizations or GPU scaling
* Basic training loop without advanced logging

---

## 🔬 Future Improvements

* Switch to subword tokenization (BPE / WordPiece)
* Add evaluation metrics (perplexity)
* Implement better training strategies (scheduler, gradient clipping)
* Scale dataset and model size
* Extend to encoder-decoder architecture

---

## 📌 Notes

* Large datasets and raw corpora are excluded for repository cleanliness
* Place custom datasets in `src/data/` to train on different corpora

---

## 🧪 Testing

```bash
python tests/test_forward.py
```

* Verifies forward pass
* Checks gradients and parameter shapes

---

## 💡 What This Project Demonstrates

* Deep understanding of Transformer internals
* Ability to implement models from first principles
* System-level thinking (data → model → training → visualization)
* Experimentation and analysis capability

---

## 🧑‍💻 Author

Built as part of a deep dive into transformer architectures and large language models.
