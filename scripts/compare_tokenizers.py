"""Char-Level vs BPE Tokenizer Comparison."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tokenizer.char_tokenizer import CharTokenizer
from tokenizer.bpe_tokenizer import BPETokenizer

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')

data_path = os.path.join(PROJECT_ROOT, "src", "data", "input.txt")
if not os.path.exists(data_path):
    data_path = os.path.join(PROJECT_ROOT, "src", "data", "input2.txt")

text = open(data_path, encoding="utf-8").read()
print(f"Dataset: {len(text):,} chars\n")

char_tok = CharTokenizer(text)
samples = [text[i:i+200] for i in range(0, min(600, len(text)), 200)]

merge_configs = [100, 300, 500]
results = [{"name": "Char", "vocab": char_tok.vocab_size,
            "avg_tokens": sum(len(char_tok.encode(s)) for s in samples)/len(samples),
            "compression": 1.0}]

for m in merge_configs:
    bpe = BPETokenizer(text, num_merges=m)
    avg = sum(len(bpe.encode(s)) for s in samples) / len(samples)
    comp = results[0]["avg_tokens"] / avg
    results.append({"name": f"BPE-{m}", "vocab": bpe.vocab_size,
                     "avg_tokens": avg, "compression": comp})
    print(f"BPE(m={m}): vocab={bpe.vocab_size}, compression={comp:.1f}x")

os.makedirs(os.path.join(PROJECT_ROOT, "experiments", "plots"), exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
names = [r["name"] for r in results]
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']

for ax, key, title in zip(axes, ["vocab", "avg_tokens", "compression"],
                           ["Vocabulary Size", "Avg Tokens/Sample", "Compression"]):
    vals = [r[key] for r in results]
    ax.bar(names, vals, color=colors[:len(names)], edgecolor='white', linewidth=1.5)
    ax.set_title(title, fontweight='bold', fontsize=14)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals)*0.02, f"{v:.1f}" if isinstance(v, float) else str(v),
                ha='center', fontweight='bold')

fig.suptitle('Tokenizer Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
path = os.path.join(PROJECT_ROOT, "experiments", "plots", "tokenizer_comparison.png")
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {path}")
