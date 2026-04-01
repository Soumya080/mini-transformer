import torch
import torch.nn as nn

from model.normalization.layer_norm import LayerNorm
from model.attention.masked_attention import MultiHeadAttention
from model.feed_forward.ffn import FeedForward


class DecoderBlock(nn.Module):
    """
    Single GPT Decoder Block
    """

    def __init__(self, d_model, d_ff, n_heads):
        super().__init__()

        # Submodules
        self.attn = MultiHeadAttention(d_model, n_heads) # type: ignore
        # self.attn = MultiHeadAttention(d_model, d_head)
        self.ln1 = LayerNorm(d_model)

        self.ffn = FeedForward(d_model, d_ff)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, T, d_model) or (T, d_model)
        """

        # Masked self-attention + residual
        attn_out = self.attn(x)
        x = x + attn_out
        x = self.ln1(x)

        # Feed-forward + residual
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.ln2(x)

        return x
