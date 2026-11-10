"""HW3: a region probe that learns how to pool the tokens under a box, instead of averaging them.

    tokens under the box (L x 3584, layer 21)
      -> pooling: "mean"  (HW2: every token weighted 1/L)
                  "attn"  (learned: per-head softmax weights over the L tokens)
      -> one 3584-d vector -> standardize -> MLP (1 or 2 hidden layers) -> 20 logits

Attention pooling, per head h (H heads, each owning a 3584/H slice of the vector):
    k_i      = W_k standardize(x_i)                       keys, 64 numbers per head
    s_{h,i}  = q_h . k_{h,i} / sqrt(64)                   one score per token and head
    a_{h,i}  = softmax over the tokens in this box of s_{h,i}
    pooled_h = sum_i a_{h,i} x_{i,h-slice}                weighted average of that head's slice

The queries q start at zero, so every score starts at 0 and attention starts as exactly the mean pool.
Any difference from HW2 is therefore learned, not an accident of initialization.

Regions have different token counts, so a batch is a flat list of tokens plus `seg`, the index of the
region each token belongs to; the softmax and the sum run per segment (no padding).
"""
from __future__ import annotations

import math

import torch
from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU}


def segment_softmax(scores: torch.Tensor, seg: torch.Tensor, n_seg: int) -> torch.Tensor:
    """Softmax of scores [T, H] within each segment (tokens of one region)."""
    H = scores.shape[1]
    idx = seg[:, None].expand(-1, H)
    m = torch.full((n_seg, H), float("-inf"), device=scores.device, dtype=scores.dtype)
    m = m.scatter_reduce(0, idx, scores, reduce="amax", include_self=True)
    e = torch.exp(scores - m[seg])
    z = torch.zeros((n_seg, H), device=scores.device, dtype=scores.dtype).index_add(0, seg, e)
    return e / z[seg]


class RegionProbe(nn.Module):
    def __init__(self, d: int = 3584, n_classes: int = 20, pooling: str = "attn", heads: int = 1, key_dim: int = 64,
                 hidden: tuple[int, ...] = (512,), activation: str = "relu", dropout: float = 0.5):
        super().__init__()
        assert pooling in ("mean", "attn") and d % heads == 0
        self.config = {"d": d, "n_classes": n_classes, "pooling": pooling, "heads": heads, "key_dim": key_dim,
                       "hidden": list(hidden), "activation": activation, "dropout": dropout}
        self.pooling, self.heads, self.key_dim = pooling, heads, key_dim
        self.register_buffer("tok_mean", torch.zeros(d))   # token statistics, for the keys
        self.register_buffer("tok_std", torch.ones(d))
        self.register_buffer("pool_mean", torch.zeros(d))  # pooled-vector statistics, for the MLP (as HW2)
        self.register_buffer("pool_std", torch.ones(d))
        if pooling == "attn":
            self.key = nn.Linear(d, heads * key_dim, bias=False)
            nn.init.normal_(self.key.weight, std=1.0 / math.sqrt(d))
            self.query = nn.Parameter(torch.zeros(heads, key_dim))  # zero: starts as the mean pool
        layers: list[nn.Module] = []
        prev = d
        for h in hidden:
            layers += [nn.Linear(prev, h), ACTIVATIONS[activation](), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.mlp = nn.Sequential(*layers)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def pool(self, tokens: torch.Tensor, seg: torch.Tensor, n_seg: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """tokens [T, d] raw -> pooled [n_seg, d] raw, plus attention weights [T, H] (None for mean pooling)."""
        T, d = tokens.shape
        if self.pooling == "mean":
            s = torch.zeros((n_seg, d), device=tokens.device, dtype=tokens.dtype).index_add(0, seg, tokens)
            n = torch.zeros(n_seg, device=tokens.device, dtype=tokens.dtype).index_add(0, seg, torch.ones_like(seg, dtype=tokens.dtype))
            return s / n[:, None], None
        k = self.key((tokens - self.tok_mean) / self.tok_std).view(T, self.heads, self.key_dim)
        scores = (k * self.query[None]).sum(-1) / math.sqrt(self.key_dim)         # [T, H]
        w = segment_softmax(scores, seg, n_seg)                                     # [T, H]
        v = tokens.view(T, self.heads, d // self.heads)                             # each head owns a slice
        pooled = torch.zeros((n_seg, self.heads, d // self.heads), device=tokens.device, dtype=tokens.dtype)
        pooled = pooled.index_add(0, seg, w[..., None] * v).reshape(n_seg, d)
        return pooled, w

    def forward_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.mlp((pooled - self.pool_mean) / self.pool_std)

    def forward(self, tokens: torch.Tensor, seg: torch.Tensor, n_seg: int) -> torch.Tensor:
        pooled, _ = self.pool(tokens, seg, n_seg)
        return self.forward_pooled(pooled)
