"""
Hyperparameter Sweep Runner

Trains multiple configurations and saves results.

Usage: python scripts/run_sweep.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
import math
import csv
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
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
vocab_size = tokenizer.vocab_size
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

max_seq_len = 64
batch_size = 16

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(0, len(d) - max_seq_len - 1, (batch_size,))
    x = torch.stack([d[i:i+max_seq_len] for i in ix])
    y = torch.stack([d[i+1:i+max_seq_len+1] for i in ix])
    return x.to(device), y.to(device)

# 
# Sweep configurations
# 
configs = [
    {"d_model": 64,  "n_heads": 2, "n_layers": 2, "d_ff": 256,  "steps": 3000},
    {"d_model": 64,  "n_heads": 4, "n_layers": 4, "d_ff": 256,  "steps": 3000},
    {"d_model": 128, "n_heads": 4, "n_layers": 4, "d_ff": 512,  "steps": 3000},
    {"d_model": 128, "n_heads": 4, "n_layers": 6, "d_ff": 512,  "steps": 3000},
    {"d_model": 256, "n_heads": 8, "n_layers": 4, "d_ff": 1024, "steps": 3000},
    {"d_model": 256, "n_heads": 8, "n_layers": 6, "d_ff": 1024, "steps": 3000},
]

os.makedirs(os.path.join(PROJECT_ROOT, "experiments", "results"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "experiments", "plots"), exist_ok=True)

results_path = os.path.join(PROJECT_ROOT, "experiments", "results", "sweep_results.csv")

all_results = []

for i, cfg in enumerate(configs):
    print(f"\n{'='*60}")
    print(f"Config {i+1}/{len(configs)}: d={cfg['d_model']}, "
          f"h={cfg['n_heads']}, L={cfg['n_layers']}")
    print(f"{'='*60}")

    model = GPT(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        d_ff=cfg["d_ff"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        max_sequence_len=max_seq_len
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for step in range(cfg["steps"]):
        x, y = get_batch("train")
        logits = model(x)
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B*T, V), y.reshape(B*T))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 1000 == 0:
            print(f"  Step {step} | Loss {loss.item():.3f}")

    # Final eval
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(30):
            x, y = get_batch("val")
            logits = model(x)
            B, T, V = logits.shape
            vl = criterion(logits.reshape(B*T, V), y.reshape(B*T))
            val_losses.append(vl.item())

    final_loss = sum(val_losses) / len(val_losses)
    ppl = math.exp(final_loss)
    print(f"  -> Val Loss: {final_loss:.3f} | Perplexity: {ppl:.2f}")

    all_results.append({**cfg, "params": params,
                        "val_loss": final_loss, "perplexity": ppl})

# Save CSV
with open(results_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["d_model", "n_heads", "n_layers",
                                            "d_ff", "steps", "params",
                                            "val_loss", "perplexity"])
    writer.writeheader()
    writer.writerows(all_results)

print(f"\n[SUCCESS] Sweep results saved to {results_path}")

# 
# Plot: Parameters vs Perplexity (log-log)
# 
params_list = [r["params"] for r in all_results]
ppl_list = [r["perplexity"] for r in all_results]

fig, ax = plt.subplots(figsize=(10, 7))
ax.loglog(params_list, ppl_list, 'o-', color='#E74C3C',
          markersize=10, linewidth=2, markeredgecolor='white', markeredgewidth=2)

for r in all_results:
    ax.annotate(f'd={r["d_model"]}\nh={r["n_heads"]}, L={r["n_layers"]}',
                (r["params"], r["perplexity"]),
                textcoords="offset points", xytext=(10, 5), fontsize=8)

ax.set_xlabel('Parameters', fontsize=13)
ax.set_ylabel('Perplexity', fontsize=13)
ax.set_title('Mini-Transformer Scaling Behavior\n'
             '(Perplexity vs Parameters — Log-Log Scale)',
             fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')

path = os.path.join(PROJECT_ROOT, "experiments", "plots", "scaling_law.png")
plt.savefig(path, dpi=150, bbox_inches='tight')
print(f"[SUCCESS] Scaling plot saved to {path}")
plt.close()
