import torch
import torch.nn as nn

from model.blocks.decoder import DecoderBlock
from model.normalization.layer_norm import LayerNorm


class GPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        d_ff,
        n_heads,
        n_layers,
        max_sequence_len
    ):
        super().__init__()

        # token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, d_model)

        # positional embeddings
        self.position_embeddings = nn.Embedding(max_sequence_len, d_model)

        # transformer blocks
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(d_model, d_ff, n_heads)
                for _ in range(n_layers)
            ]
        )

        # final layer norm
        self.ln_f = LayerNorm(d_model)

        # output projection
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):

        B, T = idx.shape

        # token embeddings
        tok_emb = self.token_embeddings(idx)

        # position ids
        pos = torch.arange(T, device=idx.device)

        # positional embeddings
        pos_emb = self.position_embeddings(pos)

        # combine
        x = tok_emb + pos_emb

        # transformer layers
        for block in self.blocks:
            x = block(x)

        # final norm
        x = self.ln_f(x)

        # output logits
        logits = self.lm_head(x)

        return logits