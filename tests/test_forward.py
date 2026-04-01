import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from model.gpt import GPT

model = GPT(
    vocab_size=65,
    d_model=128,
    d_ff=512,
    n_heads=4,
    n_layers=4,
    max_sequence_len=64
)

x = torch.randint(0, 50, (4, 32))

logits = model(x)

print("Output shape:", logits.shape)
print(model)
loss = logits.mean()

loss.backward()

for name, param in model.named_parameters():
    if param.grad is not None:
        print(name, param.grad.norm())

print("Total parameters:", sum(p.numel() for p in model.parameters()))