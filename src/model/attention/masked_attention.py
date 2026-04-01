import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads):

        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # projections
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):

        B, T, C = x.shape

        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # split heads
        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # attention
        scores = Q @ K.transpose(-2, -1)
        scores = scores / (self.head_dim ** 0.5)

        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        A = F.softmax(scores, dim=-1)

        out = A @ V

        # merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.out_proj(out)

        return out