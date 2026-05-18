"""
Mini-Transformer Demo — train and generate in ~5 minutes.
Usage: python scripts/demo.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch, torch.nn as nn, torch.optim as optim, math
from tokenizer.char_tokenizer import CharTokenizer
from model.gpt import GPT

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = os.path.join(PROJECT_ROOT, "src", "data", "input.txt")
if not os.path.exists(data_path):
    data_path = os.path.join(PROJECT_ROOT, "src", "data", "input2.txt")

text = open(data_path, encoding="utf-8").read()
tokenizer = CharTokenizer(text)
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
max_seq_len, batch_size = 64, 16

def get_batch():
    ix = torch.randint(0, len(train_data) - max_seq_len - 1, (batch_size,))
    x = torch.stack([train_data[i:i+max_seq_len] for i in ix])
    y = torch.stack([train_data[i+1:i+max_seq_len+1] for i in ix])
    return x.to(device), y.to(device)

model = GPT(vocab_size=tokenizer.vocab_size, d_model=64, d_ff=256,
            n_heads=4, n_layers=2, max_sequence_len=max_seq_len).to(device)
params = sum(p.numel() for p in model.parameters())

print(f"\n🔧 Mini-Transformer Demo (params={params:,}, device={device})")
print(f"   Training for 2000 steps...\n")

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for step in range(2000):
    x, y = get_batch()
    logits = model(x)
    B, T, V = logits.shape
    loss = criterion(logits.reshape(B*T, V), y.reshape(B*T))
    optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if step % 500 == 0:
        print(f"  Step {step:>5d} | Loss {loss.item():.3f}")

print(f"\n✅ Done! Final loss: {loss.item():.3f} | Perplexity: {math.exp(loss.item()):.2f}\n")

model.eval()
idx = torch.tensor([tokenizer.encode("The ")], device=device)
for _ in range(200):
    logits = model(idx[:, -max_seq_len:])[:, -1, :] / 0.8
    idx = torch.cat((idx, torch.multinomial(torch.softmax(logits, -1), 1)), 1)

print(f"📝 Generated:\n   {tokenizer.decode(idx[0].tolist())[:300]}")
