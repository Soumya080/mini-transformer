import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
import math
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tokenizer.char_tokenizer import CharTokenizer
from model.gpt import GPT

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')


# Hyperparameters

batch_size = 16
d_model = 128
n_heads = 4
d_ff = 512
n_layers = 4
max_seq_len = 64
num_steps = 5000
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load data

data_path = os.path.join(PROJECT_ROOT, "src", "data", "input.txt")
if not os.path.exists(data_path):
    data_path = os.path.join(PROJECT_ROOT, "src", "data", "input2.txt")

with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharTokenizer(text)
encoded = tokenizer.encode(text)
data = torch.tensor(encoded, dtype=torch.long)
vocab_size = tokenizer.vocab_size

print(f"Dataset size: {len(data):,} characters")
print(f"Vocab size: {vocab_size}")
print(f"Device: {device}")

# ========================================
# Train/Val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    dataset = train_data if split == "train" else val_data
    ix = torch.randint(0, len(dataset) - max_seq_len - 1, (batch_size,))
    x = torch.stack([dataset[i:i+max_seq_len] for i in ix])
    y = torch.stack([dataset[i+1:i+max_seq_len+1] for i in ix])
    return x.to(device), y.to(device)

# ========================================
# Model
# ========================================
model = GPT(
    vocab_size=vocab_size,
    d_model=d_model,
    d_ff=d_ff,
    n_heads=n_heads,
    n_layers=n_layers,
    max_sequence_len=max_seq_len
).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {total_params:,}")
print(f"Config: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

@torch.no_grad()
def estimate_loss(eval_iters=20):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch("val")
        logits = model(x)
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B*T, V), y.reshape(B*T))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

# ========================================
# Training loop
# ========================================
print(f"\nStarting training for {num_steps} steps...\n")

train_losses = []
val_losses = []
val_steps = []

for step in range(num_steps):
    x, y = get_batch("train")
    logits = model(x)
    B, T, V = logits.shape
    loss = criterion(logits.reshape(B*T, V), y.reshape(B*T))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    train_losses.append(loss.item())

    if step % 100 == 0:
        val_loss = estimate_loss()
        val_losses.append(val_loss)
        val_steps.append(step)
        perplexity = math.exp(val_loss)
        print(f"Step {step:>5d} | Train Loss {loss.item():.3f} | Val Loss {val_loss:.3f} | Perplexity {perplexity:.2f}")

final_val_loss = estimate_loss(eval_iters=50)
final_perplexity = math.exp(final_val_loss)
print(f"\nFINAL — Val Loss: {final_val_loss:.3f} | Perplexity: {final_perplexity:.2f}")


# Save model

torch.save(model.state_dict(), os.path.join(PROJECT_ROOT, "model.pt"))
print("Model saved to model.pt")

# Plot loss curves

os.makedirs(os.path.join(PROJECT_ROOT, "experiments", "plots"), exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(train_losses, alpha=0.3, color='#4A90D9', linewidth=0.5, label='Raw')
window = 50
if len(train_losses) > window:
    smoothed = [sum(train_losses[max(0,i-window):i+1])/min(i+1,window)
                for i in range(len(train_losses))]
    ax1.plot(smoothed, color='#2C5F8A', linewidth=2, label='Smoothed')
ax1.set_xlabel('Training Step')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(val_steps, val_losses, 'o-', color='#E74C3C', linewidth=2, markersize=4)
ax2.set_xlabel('Training Step')
ax2.set_ylabel('Validation Loss')
ax2.set_title('Validation Loss', fontweight='bold')
ax2.grid(True, alpha=0.3)

ax3 = ax2.twinx()
val_ppls = [math.exp(vl) for vl in val_losses]
ax3.plot(val_steps, val_ppls, 's--', color='#27AE60', linewidth=1.5, markersize=4)
ax3.set_ylabel('Perplexity', color='#27AE60')

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "experiments", "plots", "training_loss.png"),
            dpi=150, bbox_inches='tight')
print("Loss curves saved to experiments/plots/training_loss.png")
plt.close()

# Save results to CSV

os.makedirs(os.path.join(PROJECT_ROOT, "experiments", "results"), exist_ok=True)
results_path = os.path.join(PROJECT_ROOT, "experiments", "results", "training_results.csv")
file_exists = os.path.exists(results_path)

with open(results_path, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["n_heads", "d_model", "n_layers", "d_ff",
                         "num_steps", "val_loss", "perplexity", "params"])
    writer.writerow([n_heads, d_model, n_layers, d_ff,
                     num_steps, f"{final_val_loss:.4f}",
                     f"{final_perplexity:.2f}", total_params])
print(f"Results saved to {results_path}")

# Generate text samples

def generate(model, prompt_text, max_new_tokens=200, temperature=0.8):
    model.eval()
    tokens = tokenizer.encode(prompt_text)
    idx = torch.tensor([tokens], device=device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -max_seq_len:]
        logits = model(idx_cond)[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
    return tokenizer.decode(idx[0].tolist())

print(f"\n{'='*50}")
print("Generated Text Samples")
print(f"{'='*50}\n")

for p in ["The ", "To be", "In the"]:
    print(f'Prompt: "{p}"')
    print(f"Output: {generate(model, p, max_new_tokens=150)[:200]}")
    print("-" * 40)