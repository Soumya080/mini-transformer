# Experiments

This document summarizes the hyperparameter sweep experiments conducted on the Mini-Transformer model.

---

## Experimental Setup

All experiments were run using the character-level GPT model trained on a small text corpus (`input2.txt`). The training pipeline uses Adam optimizer with a learning rate of 1e-3 and gradient clipping at 1.0.

### Baseline Configuration

| Parameter      | Value |
| -------------- | ----- |
| d_model        | 128   |
| n_heads        | 1     |
| d_ff           | 512   |
| n_layers       | 4     |
| max_seq_len    | 64    |
| batch_size     | 16    |
| learning_rate  | 1e-3  |

---

## Hyperparameter Sweeps

### 1. Number of Attention Heads

Compared single-head vs. multi-head attention to measure the impact on convergence and validation loss.

| n_heads | Validation Loss | Perplexity |
| ------- | --------------- | ---------- |
| 1       | ~1.68           | ~5.39      |
| 2       | ~1.66           | ~5.27      |

**Observation**: Multi-head attention (n_heads=2) produced steeper early gradients and achieved a lower perplexity bound compared to a single attention head.

### 2. Model Dimension (`d_model`)

Swept across d_model ∈ {32, 64, 128} to evaluate the effect of embedding dimensionality on learning capacity.

- Attention heatmaps were generated at training steps 3,000 and 10,000 for each configuration.
- Larger `d_model` values increased representational capacity but showed diminishing returns on the small corpus due to overfitting.

### 3. Maximum Sequence Length (`max_seq_len`)

Swept across max_seq_len ∈ {32, 64, 128} to observe the impact of context window size.

- Longer context windows allowed the model to capture more distant dependencies.
- Attention heatmaps at steps 3,000 and 10,000 show progressively broader attention patterns with longer sequences.

### 4. Number of Layers (`n_layers`)

Compared n_layers ∈ {4, 6} to evaluate depth vs. performance trade-offs.

- Deeper networks showed marginal improvements but increased training time.
- Attention heatmaps confirm that additional layers refine attention distributions.

---

## Attention Visualizations

Attention heatmaps for each experiment are stored in:

```
experiments/plots/
```

Naming convention: `step_{N}_{parameter}={value}.png`

Examples:
- `step_3000_d_model=64.png` — Attention at step 3,000 with d_model=64
- `step_10000_max_len=128.png` — Attention at step 10,000 with max_seq_len=128

---

## Key Insights

1. **Multi-head attention** improves convergence speed and final loss, even on small datasets.
2. **Increasing d_model** beyond the dataset's complexity yields diminishing returns and risks overfitting.
3. **Longer context windows** help capture broader patterns but require proportionally more data to generalize.
4. **Deeper networks** refine attention but add computational overhead with marginal gains on small corpora.

---

## Results Data

Raw experiment results (CSV) are stored in:

```
experiments/results/experiments_heads.csv
```
