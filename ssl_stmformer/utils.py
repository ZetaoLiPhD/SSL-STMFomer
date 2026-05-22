"""Lightweight utilities used across SSL-STMFormer.

All helpers are kept dependency-free (numpy/scipy/torch) so the package can be
dropped into any environment that already has those.
"""

from __future__ import annotations

import math
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Scalers
# ---------------------------------------------------------------------------

class StandardScaler:
    """Per-feature standardiser; fitted on the training split only."""

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, x: np.ndarray):
        self.mean = float(np.mean(x))
        self.std = float(np.std(x))
        if self.std < 1e-8:
            self.std = 1.0
        return self

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

def laplacian_eigenvectors(adj: np.ndarray, k: int) -> np.ndarray:
    """Return the ``k`` smallest non-trivial Laplacian eigenvectors.

    Used for the spatial Graph Laplacian PE. We follow the symmetric
    normalisation: ``L = I - D^{-1/2} A D^{-1/2}``.
    """
    a = np.asarray(adj, dtype=np.float64).copy()
    a = (a + a.T) / 2.0
    np.fill_diagonal(a, 0.0)
    deg = a.sum(axis=1)
    d_inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    d_inv_sqrt[nz] = deg[nz] ** -0.5
    d_inv_sqrt = np.diag(d_inv_sqrt)
    n = a.shape[0]
    lap = np.eye(n) - d_inv_sqrt @ a @ d_inv_sqrt
    eig_val, eig_vec = np.linalg.eigh(lap)
    # Drop the trivial zero eigenvector and keep the k smallest non-trivial.
    idx = np.argsort(eig_val)
    eig_vec = eig_vec[:, idx]
    eig_vec = eig_vec[:, 1 : k + 1]
    if eig_vec.shape[1] < k:
        pad = np.zeros((n, k - eig_vec.shape[1]))
        eig_vec = np.concatenate([eig_vec, pad], axis=1)
    return eig_vec.astype(np.float32)


def hop_distance(adj: np.ndarray, max_hops: int = 511) -> np.ndarray:
    """Pairwise hop distances via vectorised Floyd-Warshall.

    Run on the unweighted adjacency. Disconnected pairs are clipped to
    ``max_hops``. The result is an ``(N, N)`` integer matrix used to
    derive the short-range mask ``M_short = 1[hop <= lambda]``.
    """
    a = (np.asarray(adj) > 0).astype(np.int32)
    np.fill_diagonal(a, 0)
    n = a.shape[0]
    sh = np.full((n, n), max_hops, dtype=np.int32)
    sh[a > 0] = 1
    np.fill_diagonal(sh, 0)
    # Vectorised Floyd-Warshall: O(N^3) but stays in C.
    for k in range(n):
        sh = np.minimum(sh, sh[:, k : k + 1] + sh[k : k + 1, :])
    sh = np.minimum(sh, max_hops)
    return sh.astype(np.int32)


def hop_distance_cached(adj: np.ndarray, cache_path: Optional[str] = None) -> np.ndarray:
    """``hop_distance`` with optional on-disk caching."""
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)
    sh = hop_distance(adj)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, sh)
    return sh


def dtw_distance_matrix(
    series: np.ndarray,
    points_per_hour: int,
    radius: int = 6,
    verbose: bool = True,
) -> np.ndarray:
    """Pairwise DTW distance using ``fastdtw`` on a daily mean profile.

    The flow is first collapsed into a length-``24*pph`` daily profile by
    averaging across all available days, which keeps the DTW cost
    tractable for thousands of nodes. ``fastdtw`` is then called for each
    unordered pair of nodes.
    """
    try:
        from fastdtw import fastdtw  # type: ignore
    except ImportError as e:  # pragma: no cover - hard dependency for real datasets
        raise RuntimeError(
            "fastdtw is required for the long-range mask. "
            "Install with `pip install fastdtw`."
        ) from e

    series = np.asarray(series, dtype=np.float32)
    if series.ndim == 2:
        series = series[..., None]
    t, n, f = series.shape
    day_len = 24 * points_per_hour
    days = max(1, t // day_len)
    used = days * day_len
    profile = series[:used].reshape(days, day_len, n, f).mean(axis=0)  # (day_len, N, F)

    sim = np.zeros((n, n), dtype=np.float32)
    rng = range(n)
    if verbose:
        try:
            from tqdm import tqdm  # type: ignore
            rng = tqdm(rng, desc="DTW")
        except ImportError:
            pass
    for i in rng:
        for j in range(i, n):
            d, _ = fastdtw(profile[:, i, :], profile[:, j, :], radius=radius)
            sim[i, j] = d
            sim[j, i] = d
    return sim


def dtw_topk_mask(dtw_matrix: np.ndarray, k: int) -> np.ndarray:
    """Top-K mask: 1 where ``j`` is in the K nearest neighbours of ``i``."""
    n = dtw_matrix.shape[0]
    mask = np.zeros((n, n), dtype=np.float32)
    order = np.argsort(dtw_matrix, axis=1)[:, : max(1, k)]
    rows = np.arange(n)[:, None].repeat(order.shape[1], axis=1)
    mask[rows, order] = 1.0
    return mask


def dtw_matrix_cached(
    series: np.ndarray,
    points_per_hour: int,
    cache_path: Optional[str] = None,
    radius: int = 6,
) -> np.ndarray:
    """``dtw_distance_matrix`` with optional on-disk caching."""
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)
    m = dtw_distance_matrix(series, points_per_hour=points_per_hour, radius=radius)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, m)
    return m


# ---------------------------------------------------------------------------
# Loss / metrics
# ---------------------------------------------------------------------------

def masked_mae(pred: torch.Tensor, true: torch.Tensor, mask_value: Optional[float] = None) -> torch.Tensor:
    if mask_value is not None:
        mask = true > mask_value
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(pred - true))


def masked_mse(pred, true, mask_value=None):
    if mask_value is not None:
        mask = true > mask_value
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean((pred - true) ** 2)


def masked_rmse(pred, true, mask_value=None):
    return torch.sqrt(masked_mse(pred, true, mask_value))


def masked_mape(pred, true, mask_value=None):
    if mask_value is not None:
        mask = true > mask_value
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs((true - pred) / true))


def metric_dict(pred: torch.Tensor, true: torch.Tensor, mask_value: float = 0.0) -> dict:
    return {
        "MAE": masked_mae(pred, true, mask_value).item(),
        "RMSE": masked_rmse(pred, true, mask_value).item(),
        "MAPE": masked_mape(pred, true, mask_value).item() * 100.0,
    }


# ---------------------------------------------------------------------------
# Dynamic Weight Average for SSL multi-task balancing
# ---------------------------------------------------------------------------

def dwa(loss_old: np.ndarray, loss_older: np.ndarray, T: float = 2.0) -> np.ndarray:
    loss_old = torch.tensor(loss_old, dtype=torch.float32)
    loss_older = torch.tensor(loss_older, dtype=torch.float32)
    n = len(loss_old)
    r = loss_old / (loss_older + 1e-8)
    return (n * F.softmax(r / T, dim=0)).numpy()


# ---------------------------------------------------------------------------
# Misc.
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
