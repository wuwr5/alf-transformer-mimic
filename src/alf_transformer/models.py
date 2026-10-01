# -*- coding: utf-8 -*-
"""Transformer architectures for tabular + longitudinal clinical data.

Two architectures are provided:

``FTTransformer``
    Feature Tokenizer + Transformer (Gorishniy et al., NeurIPS 2021). Each
    numerical feature is projected with its own weight vector and each binary
    feature with its own bias vector; a learnable ``[CLS]`` token is prepended
    and the classification head reads the encoded ``[CLS]`` state.

``HybridTransformer``
    ``FTTransformer`` extended with an additional token per follow-up day. Each
    day token is built from the standardised laboratory values of that day plus
    a missingness indicator for every value, so irregular sampling is handled by
    the model rather than by discarding patients.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["FTTransformer", "HybridTransformer", "build_model"]


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer for static tabular features."""

    def __init__(self, n_num: int, n_bin: int, d_model: int = 48, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.15, ffn: int = 128):
        super().__init__()
        self.num_w = nn.Parameter(torch.randn(n_num, d_model) * 0.02)
        self.num_b = nn.Parameter(torch.zeros(n_num, d_model))
        self.bin_w = nn.Parameter(torch.randn(max(n_bin, 1), d_model) * 0.02)
        self.bin_b = nn.Parameter(torch.zeros(max(n_bin, 1), d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ffn, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x_num, x_bin, x_traj=None, x_mask=None):
        b = x_num.shape[0]
        tokens = [x_num.unsqueeze(-1) * self.num_w.unsqueeze(0) + self.num_b.unsqueeze(0)]
        if x_bin.shape[1] > 0:
            tokens.append(x_bin.unsqueeze(-1) * self.bin_w.unsqueeze(0) + self.bin_b.unsqueeze(0))
        h = torch.cat(tokens, dim=1)
        h = torch.cat([self.cls.expand(b, -1, -1), h], dim=1)
        h = self.encoder(h)
        return self.head(self.norm(h[:, 0])).squeeze(-1)


class HybridTransformer(nn.Module):
    """Static tabular tokens plus one token per follow-up day."""

    def __init__(self, n_num: int, n_bin: int, n_days: int = 7, n_traj: int = 4,
                 d_model: int = 48, n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.15, ffn: int = 128):
        super().__init__()
        self.num_w = nn.Parameter(torch.randn(n_num, d_model) * 0.02)
        self.num_b = nn.Parameter(torch.zeros(n_num, d_model))
        self.bin_w = nn.Parameter(torch.randn(max(n_bin, 1), d_model) * 0.02)
        self.bin_b = nn.Parameter(torch.zeros(max(n_bin, 1), d_model))
        self.day_proj = nn.Linear(n_traj * 2, d_model)
        self.day_pos = nn.Parameter(torch.randn(n_days, d_model) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ffn, dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x_num, x_bin, x_traj, x_mask):
        b = x_num.shape[0]
        tokens = [x_num.unsqueeze(-1) * self.num_w.unsqueeze(0) + self.num_b.unsqueeze(0)]
        if x_bin.shape[1] > 0:
            tokens.append(x_bin.unsqueeze(-1) * self.bin_w.unsqueeze(0) + self.bin_b.unsqueeze(0))
        h = torch.cat(tokens, dim=1)
        day = torch.cat([x_traj, x_mask], dim=-1)
        h = torch.cat([h, self.day_proj(day) + self.day_pos.unsqueeze(0)], dim=1)
        h = torch.cat([self.cls.expand(b, -1, -1), h], dim=1)
        h = self.encoder(h)
        return self.head(self.norm(h[:, 0])).squeeze(-1)


def build_model(arch: str, n_num: int, n_bin: int, **kwargs) -> nn.Module:
    """Factory. ``arch`` is ``"ft"`` or ``"hybrid"``."""
    arch = arch.lower()
    if arch == "ft":
        return FTTransformer(n_num, n_bin, **kwargs)
    if arch == "hybrid":
        return HybridTransformer(n_num, n_bin, **kwargs)
    raise ValueError(f"unknown architecture: {arch!r} (expected 'ft' or 'hybrid')")
