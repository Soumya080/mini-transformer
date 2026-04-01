# Architecture of GPT-Style Mini-Transformer

## 1. Overview

This project implements a decoder-only, GPT-style transformer entirely from scratch using PyTorch. The goal of this architecture is to provide a deep, implementation-level understanding of how self-attention and transformer blocks function under the hood, tailored for autoregressive character-level language modeling.

**Key Design Decisions:**
* **Character-Level Tokenization**: Instead of sub-word or word-level splits (like BPE), text is tokenized at the character level.
* **Causal Masking**: Ensures autoregressive properties (predicting the *next* token) by blinding the model to future interactions.
* **Custom Autograd**: The Feed-Forward Network explicitly implements the forward and backward passes (`torch.autograd.Function`) instead of relying solely on PyTorch's automatic differentiation.
* **Post-Layer Normalization**: Follows the original Transformer structure (Attention → Add → LayerNorm) rather than the Pre-LN structure typical in GPT-2.

---

## 2. Architecture Breakdown

### Token & Positional Embeddings
To process discrete symbols, the input sequences of character IDs are converted into dense vector representations.
* **Token Embeddings (`nn.Embedding`)**: Maps a vocabulary of characters to vectors of size `d_model`.
* **Positional Embeddings (`nn.Embedding`)**: Injects sequential order by providing learned location embeddings for each token position up to `max_seq_len`.
* **Combination**: $X = \text{TokenEmbeddings}(idx) + \text{PositionalEmbeddings}(pos)$. This combined representation $X$ is fed into the transformer blocks.

### Multi-Head Self-Attention
This layer allows tokens to gather context from other tokens in the sequence.
* **Projections**: The input is linearly projected into Queries ($Q$), Keys ($K$), and Values ($V$) using weight matrices without biases.
* **Multi-Head Parallelization**: The vectors are reshaped and split into `n_heads`, each with dimension `head_dim = d_model / n_heads`, allowing different heads to specialize in different representational subspaces.
* **Scaled Dot-Product**: Attention scores are computed as $Q \cdot K^T$ and scaled down by $\sqrt{d_{k}}$ (`head_dim ** 0.5`) to prevent vanishing gradients during softmax.
* **Causal Masking**: A lower-triangular matrix (`torch.tril`) is applied, filling upper-triangle values with $-\infty$. This guarantees a token at position $t$ only attends to tokens at positions $\le t$.
* **Output Construction**: Softmax is applied, the result is multiplied by $V$, and the heads are concatenated and linearly projected via an output weight matrix.

### Decoder Block
The model stacks `n_layers` of Decoder Blocks.
* **Flow**: Input → Masked Self-Attention → Add (Residual) → LayerNorm → Feed-Forward → Add (Residual) → LayerNorm.
* **Residual Connections**: Skip connections (`x = x + sublayer(x)`) mitigate the vanishing gradient problem in deep networks, allowing smooth gradient flow.
* **Layer Normalization**: Stabilizes the intermediate representations, normalizing across the feature dimension (`d_model`) with learned parameters $\gamma$ and $\beta$.

### Feed Forward Network (FFN)
The FFN operates on each position independently, expanding the dimensionality.
* **Structure**: A two-layer perceptron: `Linear(d_model, d_ff)` → `ReLU` → `Linear(d_ff, d_model)`.
* **Custom Autograd**: Uniquely, the gradients for the FFN are calculated manually using a custom `torch.autograd.Function`. The `backward()` method computes gradients explicitly (e.g., handling the derivative of ReLU as `da * (h > 0)`) without relying on the PyTorch Autograd engine.

### Output Layer
* The final decoder block output is passed through a unified sequence LayerNorm (`self.ln_f`).
* A linear projection without biases (`self.lm_head`) expands `d_model` back to `vocab_size`, yielding the unnormalized logits predicting the next character distribution.

---

## 3. Data Flow

The forward pass traces sequence data systematically through the layers. 

```text
       Input Text String
              │
[Character Tokenizer (Encode)] 
              │
      Token IDs (B, T)
              │
 ┌────────────┴─────────────┐
 │  Token Embedding (B,T,C) │
 │            +             │ ◄── $X = \text{TokEmb} + \text{PosEmb}$
 │ Position Embedding (T,C) │
 └────────────┬─────────────┘
              │
    ┌─────────▼─────────┐
    │  Decoder Block 1  │
    │  [Attn → LN → FF] │ 
    └─────────┬─────────┘
              │
             ... (n_layers)
              │
    ┌─────────▼─────────┐
    │ Final Layer Norm  │
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │ Linear Projection │
    └─────────┬─────────┘
              │
       Logits (B, T, V)
```

---

## 4. Training Pipeline

The `train.py` script houses the pipeline for extracting data, feeding it to the model, and running backpropagation.
* **Tokenization & Processing**: Reads `input2.txt` and encodes the text using the custom `CharTokenizer`. The data is split 90% for training and 10% for validation.
* **Batching Strategy**: A sampler fetches `batch_size` random sequences of length `max_seq_len`. `X` is the string `i:i+max_seq_len` and `Y` comprises the character targets shifted by one `i+1:i+max_seq_len+1`.
* **Loss Function**: `CrossEntropyLoss` computes the difference between predicted character probabilities and actual targets by flattening the temporal dimension (`B*T, V`).
* **Optimization**: `Adam` is used with a learning rate of $10^{-3}$. Gradient clipping (`torch.nn.utils.clip_grad_norm_` at 1.0) is aggressively used to prevent exploding gradients and ensure optimization stability.
* **What the Model Learns**: Through stochastic iterations, it learns transition probabilities specific to the patterns in the tiny dataset, building a compressed autoregressive mapping of text.

---

## 5. Experiments

Extensive hyperparameter sweeping was conducted to observe variations in attention and layer combinations.

### Experiments Conducted
The parameter space mapped in `notes.md` features context length (32, 64), `d_model` (64, 128), target attention heads (1, 2, 4), and `n_layers` (2, 4). Detailed validation losses were periodically evaluated using a moving average over batch steps.

### Observations
Based on the experimental output files (`experiments_heads.csv`):
* **Single vs. Multi-Head (1 vs. 2 Heads)**: 
  * `n_heads = 1` converged at Validation Loss ~1.68 / Perplexity ~5.39.
  * `n_heads = 2` showed stronger convergence patterns hitting Validation Loss ~1.66 / Perplexity ~5.27.
* **Loss Dynamics**: Expanding from one to multiple attention heads observably enhanced outputs by generating steeper gradients early in the training loop and attaining a lower perplexity bound.

### Insights
* **Trade-offs**: Expanding `d_model` and `n_layers` increased learning capacity but led to diminishing returns on such a small corpus size due to slight early-stage overfitting. 
* **Expressivity**: Additional heads permitted the attention mechanism to track independent sequential dependencies (e.g., both immediate positional features and distant syntax ties).

---

## 6. Limitations

* **Character-Level Limitations**: Characters possess minimal intrinsic semantic value, meaning the model must dedicate numerous layers just to construct "words" before it can understand broader language semantics.
* **Dataset Constraints**: Operating on miniature test files (`input2.txt` ~219 bytes) heavily caps sequence abstraction. The model ultimately memorizes rather than generalizes.
* **Post-Layer Custom Training Stability**: Following Post-LN can historically cause deep transformer warmups to be relatively unstable compared to Pre-LayerNorm architecture (GPT-2 standard), though this is masked by our shallow network sizes.
* **Positional Restrictions**: Static absolute embeddings mean the network lacks context relative distance awareness scaling past `max_seq_len`.

---

## 7. Future Improvements

* **Upgraded Tokenization**: Transition from a basic `CharTokenizer` to Sub-word/Byte-Pair Encoding (BPE) (e.g., `tiktoken`) to enhance semantic density and inference speed.
* **Scale the Dataset**: Relocating data feeds to the substantially larger 1MB `input.txt` dataset to force generalization.
* **Pre-LN Transformation**: Moving Normalization to happen *before* attention and FFN blocks will enhance deeper layer gradient flow for larger setups.
* **Optimization Scheduling**: Apply learning rate cosine decay with warmup steps to smooth the trajectory out of initial suboptimal plateaus.
