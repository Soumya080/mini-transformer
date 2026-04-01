import sys
import os

# Add src/ to Python path so model imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.gpt import GPT

model = GPT(
    vocab_size=65,
    d_model=32,
    d_ff=128,
    d_head=8,
    n_layers=2,
    max_sequence_len=32
)

print(hasattr(model.blocks[0].attn, "last_attention"))