import sys
import os

# Add src/ to Python path so model/tokenizer imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import matplotlib.pyplot as plt

from tokenizer.char_tokenizer import CharTokenizer
from model.gpt import GPT

# Resolve paths relative to project root
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load dataset
# -------------------------

text = open(os.path.join(PROJECT_ROOT, "src", "data", "input.txt")).read()

tokenizer = CharTokenizer(text)

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

vocab_size = tokenizer.vocab_size

# -------------------------
# Load model
# -------------------------

model = GPT(
    vocab_size=vocab_size,
    d_model=32,
    d_ff=128,
    d_head=8,
    n_layers=2,
    max_sequence_len=32
).to(device)

model.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, "model.pt"), map_location=device))
model.eval()

# -------------------------
# Input sentence
# -------------------------

sample = "To be or not to be"

tokens = torch.tensor(
    tokenizer.encode(sample),
    dtype=torch.long
).unsqueeze(0).to(device)

# -------------------------
# Forward pass
# -------------------------

with torch.no_grad():

    logits = model(tokens)

# -------------------------
# Get attention
# -------------------------

attention = model.blocks[0].attn.last_attention[0].cpu()

# -------------------------
# Plot
# -------------------------

plt.figure(figsize=(6,6))

plt.imshow(attention, cmap="hot")

plt.colorbar()

plt.title("Attention Map")

plt.xlabel("Key position")

plt.ylabel("Query position")

plt.show()