
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Static masks (do not depend on the current batch)
# ---------------------------------------------------------------------------

def short_range_mask(hop_mx: np.ndarray, lam: int) -> np.ndarray:
    """Return ``M_short`` from the paper: 1 if hop <= lambda, else 0."""
    return (hop_mx <= lam).astype(np.float32)


def long_range_mask(dtw_mask: np.ndarray) -> np.ndarray:
    """Long-range mask is already binary (top-K by DTW); just return as float."""
    return dtw_mask.astype(np.float32)


# ---------------------------------------------------------------------------
# Random / tube / block masks (sampled fresh each forward pass)
# ---------------------------------------------------------------------------

def random_mask(shape, ratio: float, device) -> torch.Tensor:
    """Independent Bernoulli mask with ``1 - ratio`` probability of keeping."""
    keep = torch.rand(shape, device=device) > ratio
    return keep.float()


def tube_mask_spatial(b: int, n: int, ratio: float, device) -> torch.Tensor:
    """Drop a fraction of nodes for the whole batch sample.

    Returns a ``(b, n, n)`` mask where rows/cols of dropped nodes are zeroed.
    """
    keep = torch.rand(b, n, device=device) > ratio  # (b, n)
    m = keep.unsqueeze(1) & keep.unsqueeze(2)
    return m.float()


def tube_mask_temporal(b: int, t: int, ratio: float, device) -> torch.Tensor:
    keep = torch.rand(b, t, device=device) > ratio
    m = keep.unsqueeze(1) & keep.unsqueeze(2)
    return m.float()


def _vectorised_block_keep(b: int, l: int, block: int, device) -> torch.Tensor:
    """Build a ``(b, l)`` keep-mask that zeroes a contiguous block per sample.

    All operations stay on the device with no Python loop -- this is critical
    when ``b`` is in the hundreds (it grows to ``B*T`` in the spatial branch).
    """
    starts = torch.randint(0, max(1, l - block + 1), (b,), device=device)
    idx = torch.arange(l, device=device).unsqueeze(0)               # (1, l)
    in_block = (idx >= starts.unsqueeze(1)) & (idx < (starts + block).unsqueeze(1))
    return (~in_block).float()                                       # (b, l)


def block_mask_spatial(b: int, n: int, ratio: float, device) -> torch.Tensor:
    """Drop a contiguous block of ``round(ratio * n)`` nodes per sample."""
    block = max(1, int(round(ratio * n)))
    keep = _vectorised_block_keep(b, n, block, device)
    return keep.unsqueeze(1) * keep.unsqueeze(2)                     # (b, n, n)


def block_mask_temporal(b: int, t: int, ratio: float, device) -> torch.Tensor:
    block = max(1, int(round(ratio * t)))
    keep = _vectorised_block_keep(b, t, block, device)
    return keep.unsqueeze(1) * keep.unsqueeze(2)


def causal_mask(t: int, device) -> torch.Tensor:
    """Lower-triangular mask: position ``i`` can attend to ``j`` only if j <= i."""
    return torch.tril(torch.ones(t, t, device=device))


def apply_mask(attn_logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Apply a 0/1 multiplicative mask to attention logits.

    ``attn_logits`` has shape ``(B, H, L, L)``. ``mask`` is either ``(L, L)``
    (static, e.g. short / long / causal) or ``(B, L, L)`` (sample-specific,
    e.g. tube / block / random). We pad batch dims at the front and the
    head dim at position -3 so it broadcasts over heads. Positions with
    ``mask == 0`` are filled with a large negative number so softmax assigns
    them ~0 probability. We deliberately use ``-1e9`` instead of ``-inf`` so
    that fully-masked rows (which can happen when tube/block randomly drops
    all keys for a sample) do not produce ``NaN`` after softmax.
    """
    if mask is None:
        return attn_logits
    while mask.dim() < attn_logits.dim() - 1:
        mask = mask.unsqueeze(0)
    mask = mask.unsqueeze(-3)
    return attn_logits.masked_fill(mask == 0, -1e9)
