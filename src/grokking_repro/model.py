from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def keep_abs_topk(x: torch.Tensor, keep_fraction: float | None) -> torch.Tensor:
    if keep_fraction is None or keep_fraction >= 1.0:
        return x
    if keep_fraction <= 0.0:
        return torch.zeros_like(x)
    k = max(1, int(keep_fraction * x.shape[-1]))
    _, indices = torch.topk(x.abs(), k, dim=-1, sorted=False)
    out = torch.zeros_like(x)
    return out.scatter(-1, indices, x.gather(-1, indices))


class ActivationSparsifier:
    def __init__(self, keep_fraction: float | None = None, locations: str = "") -> None:
        self.keep_fraction = keep_fraction
        self.locations = {loc.strip() for loc in locations.split(",") if loc.strip()}

    def __call__(self, x: torch.Tensor, location: str) -> torch.Tensor:
        if location not in self.locations:
            return x
        return keep_abs_topk(x, self.keep_fraction)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        causal: bool = True,
        activation_sparsifier: ActivationSparsifier | None = None,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        self.activation_sparsifier = activation_sparsifier or ActivationSparsifier()
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        x = self.activation_sparsifier(x, "attn_in")
        q = self.activation_sparsifier(self.W_Q(x), "attn_q")
        k = self.activation_sparsifier(self.W_K(x), "attn_k")
        v = self.activation_sparsifier(self.W_V(x), "attn_v")
        q = q.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)
        if self.causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        out = self.W_O(out)
        return self.activation_sparsifier(out, "attn_out")


class SparseAwareMLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        activation_sparsifier: ActivationSparsifier | None = None,
    ) -> None:
        super().__init__()
        self.activation_sparsifier = activation_sparsifier or ActivationSparsifier()
        self.fc_in = nn.Linear(d_model, d_mlp)
        self.fc_out = nn.Linear(d_mlp, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation_sparsifier(x, "mlp_in")
        x = F.relu(self.fc_in(x))
        x = self.activation_sparsifier(x, "mlp_neuron")
        x = self.fc_out(x)
        return self.activation_sparsifier(x, "mlp_out")


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        causal: bool = True,
        activation_sparsifier: ActivationSparsifier | None = None,
    ) -> None:
        super().__init__()
        self.activation_sparsifier = activation_sparsifier or ActivationSparsifier()
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            causal=causal,
            activation_sparsifier=self.activation_sparsifier,
        )
        self.mlp = SparseAwareMLP(
            d_model=d_model,
            d_mlp=d_mlp,
            activation_sparsifier=self.activation_sparsifier,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = self.activation_sparsifier(x, "resid_post_attn")
        x = x + self.mlp(x)
        x = self.activation_sparsifier(x, "resid_post_mlp")
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
        activation_keep_fraction: float | None = None,
        activation_sparsity_locations: str = "",
    ) -> None:
        super().__init__()
        self.modulus = modulus
        self.vocab_size = modulus + 1
        self.seq_len = seq_len
        activation_sparsifier = ActivationSparsifier(
            keep_fraction=activation_keep_fraction,
            locations=activation_sparsity_locations,
        )

        self.token_embed = nn.Embedding(self.vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.empty(seq_len, d_model))
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_mlp=d_mlp,
                    causal=causal,
                    activation_sparsifier=activation_sparsifier,
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
