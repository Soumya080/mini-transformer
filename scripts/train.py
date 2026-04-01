import sys
import os

# Add src/ to Python path so model/tokenizer imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
import math 
import csv
from tokenizer.char_tokenizer import CharTokenizer
from model.gpt import GPT

# Resolve paths relative to project root
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')

print("Current working directory:", os.getcwd())
# -----------------------------
# PARAMETERS
# -----------------------------
batch_size = 16
d_model = 128
n_heads = 1
d_ff = 512
n_layers = 4
max_seq_len = 64
num_steps = 10000
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD DATA
# -----------------------------

with open(os.path.join(PROJECT_ROOT, "src", "data", "input2.txt"), "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharTokenizer(text)

encoded = tokenizer.encode(text)

data = torch.tensor(encoded, dtype=torch.long)

vocab_size = tokenizer.vocab_size

print("Dataset size:", len(data))
print("Vocab size:", vocab_size)

# -----------------------------
# TRAIN / VALIDATION SPLIT
# -----------------------------

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

# -----------------------------
# BATCH SAMPLER
# -----------------------------

def get_batch(split):

    dataset = train_data if split == "train" else val_data

    ix = torch.randint(
        0,
        len(dataset) - max_seq_len - 1,
        (batch_size,)
    )

    x = torch.stack([
        dataset[i:i+max_seq_len] for i in ix
    ])

    y = torch.stack([
        dataset[i+1:i+max_seq_len+1] for i in ix
    ])

    return x.to(device), y.to(device)

# -----------------------------
# MODEL
# -----------------------------

model = GPT(
    vocab_size=vocab_size,
    d_model=d_model,
    d_ff=d_ff,
    n_heads=n_heads,
    n_layers=n_layers,
    max_sequence_len=max_seq_len
).to(device)
print("\nModel parameters:",
      sum(p.numel() for p in model.parameters()))

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()

# -----------------------------
# VALIDATION LOSS
# -----------------------------

@torch.no_grad()
def estimate_loss():

    model.eval()

    losses = []

    for _ in range(20):

        x, y = get_batch("val")

        logits = model(x)

        B, T, V = logits.shape

        loss = criterion(
            logits.reshape(B*T, V),
            y.reshape(B*T)
        )

        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)

# -----------------------------
# TRAINING LOOP
# -----------------------------
print("NUM HEADS",n_heads)
print("NUM_LAYERS",n_layers)
print("MODEL DIMENSION",d_model)
print("TRAINING STEPS",num_steps)
print("\nStarting training...\n")

for step in range(num_steps):

    x, y = get_batch("train")

    logits = model(x)

    B, T, V = logits.shape

    loss = criterion(
        logits.reshape(B*T, V),
        y.reshape(B*T)
    )

    optimizer.zero_grad()

    loss.backward()

    # gradient clipping (stabilizes training)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    if step % 100 == 0:

        val_loss = estimate_loss()

        print(
            f"Step {step} | Train Loss {loss.item():.3f} | Val Loss {val_loss:.3f}"
        )
    # if step % 1000 == 0:
    #     torch.save(model.state_dict(), "model.pt")
# -----------------------------
# TEXT GENERATION
# -----------------------------

def generate(model, start_token, max_new_tokens):

    model.eval()

    idx = torch.tensor([[start_token]], device=device)

    for _ in range(max_new_tokens):

        # keep only last context window
        idx_cond = idx[:, -max_seq_len:]

        logits = model(idx_cond)

        logits = logits[:, -1, :]

        probs = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, 1)

        idx = torch.cat((idx, next_token), dim=1)

    return idx

print("\n---- Generated Text ----\n")

tokens = generate(model, start_token=0, max_new_tokens=300)

print(tokenizer.decode(tokens[0].tolist()))

perplexity = math.exp(val_loss)

perplexity = math.exp(val_loss)

with open(os.path.join(PROJECT_ROOT, "experiments", "results", "heads_experiment.csv"), "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        1,
        d_model,
        n_layers,
        max_seq_len,
        val_loss,
        perplexity
    ])

# # SAVE MODEL
# torch.save(model.state_dict(), "model.pt")
# print("\nModel saved as model.pt")