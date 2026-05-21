from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, causal: bool = True) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        q = self.W_Q(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_K(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_V(x).view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)
        if self.causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.W_O(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int, causal: bool = True) -> None:
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model=d_model, n_heads=n_heads, causal=causal)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.ReLU(),
            nn.Linear(d_mlp, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class ModularAdditionTransformer(nn.Module):
    def __init__(
        self,
        modulus: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_mlp: int = 512,
        n_layers: int = 1,
        seq_len: int = 3,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.modulus = modulus
        self.vocab_size = modulus + 1
        self.seq_len = seq_len

        self.token_embed = nn.Embedding(self.vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.empty(seq_len, d_model))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_mlp=d_mlp,
                    causal=causal,
                )
                for _ in range(n_layers)
            ]
        )
        self.unembed = nn.Linear(d_model, modulus, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.unembed.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(tokens) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.unembed(x[:, -1, :])

