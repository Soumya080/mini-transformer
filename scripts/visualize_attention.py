import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tokenizer.char_tokenizer import CharTokenizer
from model.gpt import GPT

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load data
data_path = os.path.join(PROJECT_ROOT, "src", "data", "input.txt")
if not os.path.exists(data_path):
    data_path = os.path.join(PROJECT_ROOT, "src", "data", "input2.txt")

text = open(data_path, encoding="utf-8").read()
tokenizer = CharTokenizer(text)
vocab_size = tokenizer.vocab_size

# Load model — MUST match training config
model = GPT(
    vocab_size=vocab_size, d_model=128, d_ff=512,
    n_heads=4, n_layers=4, max_sequence_len=64
).to(device)

checkpoint = os.path.join(PROJECT_ROOT, "model.pt")
if not os.path.exists(checkpoint):
    print("ERROR: model.pt not found! Run train.py first.")
    sys.exit(1)

model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()
print("Model loaded from model.pt")

# Input
sample = "To be or not to be"
token_ids = tokenizer.encode(sample)
token_chars = list(sample)
tokens = torch.tensor([token_ids], dtype=torch.long).to(device)

with torch.no_grad():
    logits = model(tokens)

os.makedirs(os.path.join(PROJECT_ROOT, "experiments", "plots"), exist_ok=True)

# Plot 1: Single head
attn = model.blocks[0].attn.last_attention[0]  # (n_heads, T, T)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(attn[0].cpu().numpy(), cmap='viridis', aspect='auto')
ax.set_xticks(range(len(token_chars)))
ax.set_yticks(range(len(token_chars)))
ax.set_xticklabels(token_chars, fontsize=10)
ax.set_yticklabels(token_chars, fontsize=10)
ax.set_xlabel('Key Position')
ax.set_ylabel('Query Position')
ax.set_title(f'Attention — Layer 0, Head 0\nInput: "{sample}"', fontweight='bold')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "experiments", "plots", "attention_single_head.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: attention_single_head.png")

# Plot 2: All heads comparison
n_h = attn.shape[0]
fig, axes = plt.subplots(1, n_h, figsize=(5*n_h, 5))
if n_h == 1: axes = [axes]
for h in range(n_h):
    ax = axes[h]
    ax.imshow(attn[h].cpu().numpy(), cmap='viridis', aspect='auto')
    ax.set_title(f'Head {h}', fontweight='bold')
    ax.set_xticks(range(len(token_chars)))
    ax.set_yticks(range(len(token_chars)))
    ax.set_xticklabels(token_chars, fontsize=7)
    ax.set_yticklabels(token_chars, fontsize=7)
fig.suptitle(f'Multi-Head Attention — Layer 0\nInput: "{sample}"', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "experiments", "plots", "attention_all_heads.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: attention_all_heads.png")

# Plot 3: Across layers
n_l = len(model.blocks)
fig, axes = plt.subplots(1, n_l, figsize=(5*n_l, 5))
if n_l == 1: axes = [axes]
for l in range(n_l):
    ax = axes[l]
    layer_attn = model.blocks[l].attn.last_attention[0]
    ax.imshow(layer_attn[0].cpu().numpy(), cmap='viridis', aspect='auto')
    ax.set_title(f'Layer {l}', fontweight='bold')
    ax.set_xticks(range(len(token_chars)))
    ax.set_yticks(range(len(token_chars)))
    ax.set_xticklabels(token_chars, fontsize=7)
    ax.set_yticklabels(token_chars, fontsize=7)
fig.suptitle(f'Attention Across Layers — Head 0\nInput: "{sample}"', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "experiments", "plots", "attention_across_layers.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: attention_across_layers.png")
print("\nAll attention plots saved to experiments/plots/")