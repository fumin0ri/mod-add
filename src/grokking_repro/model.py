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
        self.capture = None
        self.mean_ablation = None

    def __call__(self, x: torch.Tensor, location: str) -> torch.Tensor:
        base_location = location.split(".")[-1]
        if location in self.locations or base_location in self.locations:
            x = keep_abs_topk(x, self.keep_fraction)
        if self.capture is not None:
            self.capture(location, x)
        if self.mean_ablation is not None:
            x = self.mean_ablation(location, x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int | None = None,
        causal: bool = True,
        activation_sparsifier: ActivationSparsifier | None = None,
        attention_sink: bool = False,
        location_prefix: str = "",
    ) -> None:
        super().__init__()
        if d_head is None and d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = n_heads
        self.d_head = d_head if d_head is not None else d_model // n_heads
        self.causal = causal
        self.attention_sink = attention_sink
        self.location_prefix = location_prefix
        self.activation_sparsifier = activation_sparsifier or ActivationSparsifier()
        attn_dim = n_heads * self.d_head
        self.W_Q = nn.Linear(d_model, attn_dim, bias=True)
        self.W_K = nn.Linear(d_model, attn_dim, bias=True)
        self.W_V = nn.Linear(d_model, attn_dim, bias=True)
        self.W_O = nn.Linear(attn_dim, d_model, bias=True)
        if attention_sink:
            self.sink_logit = nn.Parameter(torch.zeros(n_heads))

    def loc(self, name: str) -> str:
        return f"{self.location_prefix}.{name}" if self.location_prefix else name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        attn_dim = self.n_heads * self.d_head
        x = self.activation_sparsifier(x, self.loc("attn_in"))
        q = self.activation_sparsifier(self.W_Q(x), self.loc("attn_q"))
        k = self.activation_sparsifier(self.W_K(x), self.loc("attn_k"))
        v = self.activation_sparsifier(self.W_V(x), self.loc("attn_v"))
        q = q.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)
        if self.causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float("-inf"))
        if self.attention_sink:
            sink_scores = self.sink_logit.view(1, self.n_heads, 1, 1).expand(batch, -1, seq_len, -1)
            scores = torch.cat([sink_scores, scores], dim=-1)
            zero_v = torch.zeros(batch, self.n_heads, 1, self.d_head, dtype=v.dtype, device=v.device)
            v = torch.cat([zero_v, v], dim=2)
        attn = F.softmax(scores, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, attn_dim)
        out = self.W_O(out)
        return self.activation_sparsifier(out, self.loc("attn_out"))


class SparseAwareMLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        activation_type: str = "relu",
        activation_sparsifier: ActivationSparsifier | None = None,
        location_prefix: str = "",
    ) -> None:
        super().__init__()
        self.activation_sparsifier = activation_sparsifier or ActivationSparsifier()
        self.location_prefix = location_prefix
        self.fc_in = nn.Linear(d_model, d_mlp)
        self.fc_out = nn.Linear(d_mlp, d_model)
        if activation_type not in {"relu", "gelu"}:
            raise ValueError("activation_type must be 'relu' or 'gelu'.")
        self.activation_type = activation_type

    def loc(self, name: str) -> str:
        return f"{self.location_prefix}.{name}" if self.location_prefix else name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation_sparsifier(x, self.loc("mlp_in"))
        x = self.fc_in(x)
        x = F.relu(x) if self.activation_type == "relu" else F.gelu(x)
        x = self.activation_sparsifier(x, self.loc("mlp_neuron"))
        x = self.fc_out(x)
        return self.activation_sparsifier(x, self.loc("mlp_out"))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        d_head: int | None = None,
        causal: bool = True,
        activation_type: str = "relu",
        activation_sparsifier: ActivationSparsifier | None = None,
        rms_norm: bool = False,
        attention_sink: bool = False,
        layer_index: int | None = None,
    ) -> None:
        super().__init__()
        self.activation_sparsifier = activation_sparsifier or ActivationSparsifier()
        self.rms_norm = rms_norm
        self.layer_prefix = f"blocks.{layer_index}" if layer_index is not None else ""
        self.ln_1 = nn.RMSNorm(d_model) if rms_norm else nn.Identity()
        self.ln_2 = nn.RMSNorm(d_model) if rms_norm else nn.Identity()
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            d_head=d_head,
            causal=causal,
            activation_sparsifier=self.activation_sparsifier,
            attention_sink=attention_sink,
            location_prefix=self.layer_prefix,
        )
        self.mlp = SparseAwareMLP(
            d_model=d_model,
            d_mlp=d_mlp,
            activation_type=activation_type,
            activation_sparsifier=self.activation_sparsifier,
            location_prefix=self.layer_prefix,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = self.activation_sparsifier(x, f"{self.layer_prefix}.resid_post_attn" if self.layer_prefix else "resid_post_attn")
        x = x + self.mlp(self.ln_2(x))
        x = self.activation_sparsifier(x, f"{self.layer_prefix}.resid_post_mlp" if self.layer_prefix else "resid_post_mlp")
        return x


class ModularAdditionTransformer(nn.Module):
    def __init__(
        self,
        modulus: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_mlp: int = 512,
        n_layers: int = 1,
        d_head: int | None = None,
        seq_len: int = 3,
        causal: bool = True,
        activation_type: str = "relu",
        activation_keep_fraction: float | None = None,
        activation_sparsity_locations: str = "",
        rms_norm: bool = False,
        use_pos_embed: bool = True,
        attention_sink: bool = False,
        bigram_table: bool = False,
    ) -> None:
        super().__init__()
        self.modulus = modulus
        self.vocab_size = modulus + 1
        self.seq_len = seq_len
        self.use_pos_embed = use_pos_embed
        self.bigram_table_enabled = bigram_table
        activation_sparsifier = ActivationSparsifier(
            keep_fraction=activation_keep_fraction,
            locations=activation_sparsity_locations,
        )
        self.activation_sparsifier = activation_sparsifier

        self.token_embed = nn.Embedding(self.vocab_size, d_model)
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.empty(seq_len, d_model))
        else:
            self.register_parameter("pos_embed", None)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_mlp=d_mlp,
                    d_head=d_head,
                    causal=causal,
                    activation_type=activation_type,
                    activation_sparsifier=activation_sparsifier,
                    rms_norm=rms_norm,
                    attention_sink=attention_sink,
                    layer_index=layer_index,
                )
                for layer_index in range(n_layers)
            ]
        )
        self.unembed = nn.Linear(d_model, modulus, bias=False)
        if bigram_table:
            self.bigram_table = nn.Parameter(torch.zeros(self.vocab_size, modulus))
        else:
            self.register_parameter("bigram_table", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embed.weight, std=0.02)
        if self.pos_embed is not None:
            nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.unembed.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(tokens)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        logits = self.unembed(x[:, -1, :])
        if self.bigram_table is not None:
            logits = logits + self.bigram_table[tokens[:, -1]]
        return logits
