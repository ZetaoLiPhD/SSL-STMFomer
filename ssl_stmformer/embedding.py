

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TemporalPositionalEncoding(nn.Module):
    """Standard sinusoidal PE along the time-step dimension."""

    def __init__(self, embed_dim: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float) * -(math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe)

    def forward(self, t: int) -> torch.Tensor:
        # Returns (1, T, 1, D) so it broadcasts against (B, T, N, D).
        return self.pe[:t].unsqueeze(0).unsqueeze(2)


class LaplacianPE(nn.Module):
    """Project the precomputed Laplacian eigenvectors onto the embedding dim."""

    def __init__(self, lape_dim: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(lape_dim, embed_dim)

    def forward(self, lap_mx: torch.Tensor) -> torch.Tensor:
        # lap_mx: (N, K) -> (1, 1, N, D)
        return self.proj(lap_mx).unsqueeze(0).unsqueeze(0)


class DataEmbedding(nn.Module):
    """The full embedding stack described above."""

    def __init__(
        self,
        feature_dim: int,
        embed_dim: int,
        lape_dim: int,
        add_time_in_day: bool = True,
        add_day_in_week: bool = True,
        dropout: float = 0.0,
        max_len: int = 1024,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week

        self.value_embed = TokenEmbedding(feature_dim, embed_dim)
        self.pos_embed = TemporalPositionalEncoding(embed_dim, max_len=max_len)
        self.lap_pe = LaplacianPE(lape_dim, embed_dim)

        if add_time_in_day:
            self.tod_embed = nn.Embedding(1440, embed_dim)
        if add_day_in_week:
            self.dow_embed = nn.Embedding(7, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        lap_mx: torch.Tensor,
        time_in_day: Optional[torch.Tensor] = None,
        day_in_week: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``x``: (B, T, N, F).

        ``time_in_day``: optional ``(B, T)`` long tensor in [0, 1440).
        ``day_in_week``: optional ``(B, T)`` long tensor in [0, 7).
        """
        b, t, n, _ = x.shape
        out = self.value_embed(x[..., : self.feature_dim])
        out = out + self.pos_embed(t).to(out.device)
        out = out + self.lap_pe(lap_mx)
        if self.add_time_in_day and time_in_day is not None:
            tod = self.tod_embed(time_in_day.long().clamp(0, 1439))  # (B, T, D)
            out = out + tod.unsqueeze(2)
        if self.add_day_in_week and day_in_week is not None:
            dow = self.dow_embed(day_in_week.long().clamp(0, 6))  # (B, T, D)
            out = out + dow.unsqueeze(2)
        return self.dropout(out)
