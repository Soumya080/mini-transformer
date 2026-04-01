import torch

with open("input.txt","r",encoding="utf-8") as f:
    text = f.read()

# unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# mappings
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text),dtype=torch.long)

# train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print("dataset length:",len(data))
print("vocab size:",vocab_size)