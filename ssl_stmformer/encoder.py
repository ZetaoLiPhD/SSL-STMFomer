
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .masks import (
    apply_mask,
    block_mask_spatial,
    block_mask_temporal,
    causal_mask,
    random_mask,
    tube_mask_spatial,
    tube_mask_temporal,
)


# ---------------------------------------------------------------------------
# Single attention head-group
# ---------------------------------------------------------------------------

class _HeadGroup(nn.Module):
    """A QKV projection + multi-head attention for one masking strategy.

    Parameters
    ----------
    embed_dim : int
        Width of the layer input.
    head_dim  : int
        Dimensionality of each attention head.
    n_heads   : int
        Number of heads in this group.
    """

    def __init__(self, embed_dim: int, head_dim: int, n_heads: int, attn_dropout: float = 0.0):
        super().__init__()
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.out_dim = head_dim * n_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(embed_dim, self.out_dim, bias=False)
        self.k = nn.Linear(embed_dim, self.out_dim, bias=False)
        self.v = nn.Linear(embed_dim, self.out_dim, bias=False)
        self.drop = nn.Dropout(attn_dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        # (..., L, out_dim) -> (..., n_heads, L, head_dim)
        x = x.view(*x.shape[:-1], self.n_heads, self.head_dim)
        return x.transpose(-3, -2)

    def attend(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """``x``: (B, L, D). Returns (B, L, head_dim * n_heads)."""
        q = self._split(self.q(x))  # (B, H, L, Dh)
        k = self._split(self.k(x))
        v = self._split(self.v(x))
        logits = (q @ k.transpose(-2, -1)) * self.scale
        logits = apply_mask(logits, mask)
        attn = logits.softmax(dim=-1)
        attn = self.drop(attn)
        out = attn @ v  # (B, H, L, Dh)
        out = out.transpose(-3, -2).contiguous().view(*x.shape[:-1], self.out_dim)
        return out


# ---------------------------------------------------------------------------
# Spatio-temporal multi-mask attention block
# ---------------------------------------------------------------------------

@dataclass
class STAttentionConfig:
    embed_dim: int
    head_dim: int = 8
    spatial_heads_per_kind: int = 2     # heads for each of {rand, tube, block, short, long}
    temporal_heads_per_kind: int = 2    # heads for each of {rand, tube, block, temp, plain}
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    mask_ratio: float = 0.1             # ratio used by random/tube/block strategies
    use_spatial_strategies: Sequence[str] = ("rand", "tube", "block", "short", "long")
    use_temporal_strategies: Sequence[str] = ("rand", "tube", "block", "temp", "plain")


class STMultiMaskAttention(nn.Module):
    """Multi-mask spatial + temporal attention with concat-fusion."""

    def __init__(self, cfg: STAttentionConfig):
        super().__init__()
        self.cfg = cfg

        self.spatial_heads = nn.ModuleDict({
            name: _HeadGroup(cfg.embed_dim, cfg.head_dim, cfg.spatial_heads_per_kind, cfg.attn_dropout)
            for name in cfg.use_spatial_strategies
        })
        self.temporal_heads = nn.ModuleDict({
            name: _HeadGroup(cfg.embed_dim, cfg.head_dim, cfg.temporal_heads_per_kind, cfg.attn_dropout)
            for name in cfg.use_temporal_strategies
        })

        total_dim = (
            len(cfg.use_spatial_strategies) * cfg.spatial_heads_per_kind
            + len(cfg.use_temporal_strategies) * cfg.temporal_heads_per_kind
        ) * cfg.head_dim
        self.proj = nn.Linear(total_dim, cfg.embed_dim)
        self.proj_drop = nn.Dropout(cfg.proj_dropout)

    # -- spatial dispatch --------------------------------------------------

    def _spatial_mask(self, name: str, b: int, n: int, device, *, short_mask, long_mask) -> Optional[torch.Tensor]:
        ratio = self.cfg.mask_ratio
        if name == "rand":
            return random_mask((b, n, n), ratio, device)
        if name == "tube":
            return tube_mask_spatial(b, n, ratio, device)
        if name == "block":
            return block_mask_spatial(b, n, ratio, device)
        if name == "short":
            return short_mask
        if name == "long":
            return long_mask
        raise ValueError(f"Unknown spatial strategy: {name}")

    def _temporal_mask(self, name: str, b: int, t: int, device) -> Optional[torch.Tensor]:
        ratio = self.cfg.mask_ratio
        if name == "rand":
            return random_mask((b, t, t), ratio, device)
        if name == "tube":
            return tube_mask_temporal(b, t, ratio, device)
        if name == "block":
            return block_mask_temporal(b, t, ratio, device)
        if name == "temp":
            return causal_mask(t, device)
        if name == "plain":
            return None
        raise ValueError(f"Unknown temporal strategy: {name}")

    # -- forward -----------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        short_mask: Optional[torch.Tensor],
        long_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """``x``: (B, T, N, D). Returns same shape."""
        b, t, n, d = x.shape
        device = x.device

        outs: List[torch.Tensor] = []

        # --- spatial branch: attend across N for each time step ------
        x_spatial = x.reshape(b * t, n, d)
        for name, group in self.spatial_heads.items():
            mask = self._spatial_mask(
                name, b * t if name in {"rand", "tube", "block"} else 1, n, device,
                short_mask=short_mask, long_mask=long_mask,
            )
            # The static masks are (N, N); broadcast over the b*t batch.
            out = group.attend(x_spatial, mask)
            out = out.view(b, t, n, group.out_dim)
            outs.append(out)

        # --- temporal branch: attend across T for each node ----------
        # Re-shape so the "sequence" axis is T.
        x_temporal = x.transpose(1, 2).reshape(b * n, t, d)
        for name, group in self.temporal_heads.items():
            if name in {"rand", "tube", "block"}:
                mask = self._temporal_mask(name, b * n, t, device)
            else:
                mask = self._temporal_mask(name, 1, t, device)
            out = group.attend(x_temporal, mask)
            out = out.view(b, n, t, group.out_dim).transpose(1, 2)
            outs.append(out)

        fused = torch.cat(outs, dim=-1)  # (B, T, N, total_dim)
        return self.proj_drop(self.proj(fused))


# ---------------------------------------------------------------------------
# Encoder block (attention + FFN, residual + LayerNorm, pre-norm style)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(embed_dim * hidden_mult)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class STEncoderLayer(nn.Module):
    def __init__(
        self,
        cfg: STAttentionConfig,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.attn = STMultiMaskAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
        self.ffn = FeedForward(cfg.embed_dim, mlp_ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        short_mask: Optional[torch.Tensor],
        long_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), short_mask, long_mask)
        x = x + self.ffn(self.norm2(x))
        return x
