"""Data loading utilities.

Three on-disk formats are supported:

1. **NPZ format** -- one file per split (``train.npz`` / ``val.npz`` /
   ``test.npz``) containing ``x``, ``y`` arrays of shape ``(B, T, N, C)``,
   plus an ``adj_mx.npz`` for the graph. Convenient when you already have
   the dataset pre-windowed.

2. **LibCity format** -- ``<name>.dyna`` (graph) or ``<name>.grid`` (grid)
   together with ``<name>.geo`` and an optional ``<name>.rel`` file. We
   provide a lightweight parser and sliding-window builder for both.

3. **Synthetic** flow data, useful for end-to-end smoke tests when no real
   data is available locally.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Sliding-window dataset
# ---------------------------------------------------------------------------

class SlidingDataset(Dataset):
    """Wrap pre-built ``x``/``y`` numpy arrays plus optional time features."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        time_in_day: Optional[np.ndarray] = None,
        day_in_week: Optional[np.ndarray] = None,
    ):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.tod = time_in_day if time_in_day is not None else None
        self.dow = day_in_week if day_in_week is not None else None

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "X": torch.from_numpy(self.x[idx]),
            "y": torch.from_numpy(self.y[idx]),
        }
        if self.tod is not None:
            item["time_in_day"] = torch.from_numpy(self.tod[idx]).long()
        if self.dow is not None:
            item["day_in_week"] = torch.from_numpy(self.dow[idx]).long()
        return item


# ---------------------------------------------------------------------------
# Per-source loaders
# ---------------------------------------------------------------------------

def load_npz(data_dir: str, dataset: str) -> Tuple[Dict[str, Dict[str, np.ndarray]], np.ndarray]:
    """Load pre-windowed ``.npz`` splits and an adjacency matrix.

    Expected layout::

        <data_dir>/<dataset>/train.npz  (x, y)
        <data_dir>/<dataset>/val.npz    (x, y)
        <data_dir>/<dataset>/test.npz   (x, y)
        <data_dir>/<dataset>/adj_mx.npz (adj_mx)
    """
    out = {}
    for split in ("train", "val", "test"):
        npz = np.load(os.path.join(data_dir, dataset, f"{split}.npz"))
        out[split] = {"x": npz["x"], "y": npz["y"]}
    adj = np.load(os.path.join(data_dir, dataset, "adj_mx.npz"))["adj_mx"].astype(np.float32)
    return out, adj


def load_libcity_dyna(
    data_dir: str,
    dataset: str,
    data_col=("traffic_flow",),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Parse a LibCity-style graph dataset (``.dyna`` + ``.geo`` + ``.rel``).

    The ``.dyna`` file is read with pandas and pivoted into a dense
    ``(T, N, F)`` tensor. The adjacency matrix is constructed from ``.rel``.

    Returns ``(flow, timestamps, adj, time_interval_seconds)``.
    """
    if isinstance(data_col, str):
        data_col = (data_col,)

    base = os.path.join(data_dir, dataset)
    geo = pd.read_csv(os.path.join(base, f"{dataset}.geo"))
    geo_ids = list(geo["geo_id"])
    n = len(geo_ids)
    id2idx = {g: i for i, g in enumerate(geo_ids)}

    usecols = ["time", "entity_id", *data_col]
    dyna = pd.read_csv(os.path.join(base, f"{dataset}.dyna"), usecols=usecols)
    dyna["nidx"] = dyna["entity_id"].map(id2idx)
    dyna = dyna.dropna(subset=["nidx"]).copy()
    dyna["nidx"] = dyna["nidx"].astype(np.int32)

    times = np.sort(dyna["time"].unique())
    t = len(times)
    time_to_idx = pd.Series(np.arange(t), index=times)
    dyna["tidx"] = dyna["time"].map(time_to_idx).astype(np.int32)

    flow = np.zeros((t, n, len(data_col)), dtype=np.float32)
    for fi, col in enumerate(data_col):
        flow[dyna["tidx"].to_numpy(), dyna["nidx"].to_numpy(), fi] = dyna[col].to_numpy(dtype=np.float32)

    adj = np.zeros((n, n), dtype=np.float32)
    rel_path = os.path.join(base, f"{dataset}.rel")
    if os.path.exists(rel_path):
        rel = pd.read_csv(rel_path, usecols=["origin_id", "destination_id"])
        rel["u"] = rel["origin_id"].map(id2idx)
        rel["v"] = rel["destination_id"].map(id2idx)
        rel = rel.dropna(subset=["u", "v"])
        adj[rel["u"].astype(int), rel["v"].astype(int)] = 1.0
        adj[rel["v"].astype(int), rel["u"].astype(int)] = 1.0

    ts = pd.to_datetime(times)
    delta = int((ts[1] - ts[0]).total_seconds()) if len(ts) > 1 else 300
    return flow, ts, adj, delta


def load_libcity_grid(
    data_dir: str,
    dataset: str,
    data_col=("inflow", "outflow"),
    connectivity: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Parse a LibCity-style grid dataset (``.grid`` + ``.geo``).

    The grid is flattened into nodes indexed by ``row * num_cols + col`` and
    a regular grid adjacency (4- or 8-neighbour) is constructed. Returns
    ``(flow, timestamps, adj, time_interval_seconds)``.
    """
    if isinstance(data_col, str):
        data_col = (data_col,)

    base = os.path.join(data_dir, dataset)
    geo = pd.read_csv(os.path.join(base, f"{dataset}.geo"))
    n_rows = int(geo["row_id"].max()) + 1
    n_cols = int(geo["column_id"].max()) + 1
    n = n_rows * n_cols

    grid = pd.read_csv(
        os.path.join(base, f"{dataset}.grid"),
        usecols=["time", "row_id", "column_id", *data_col],
    )
    times = np.sort(grid["time"].unique())
    t = len(times)
    time_to_idx = pd.Series(np.arange(t), index=times)

    grid["tidx"] = grid["time"].map(time_to_idx).astype(np.int32)
    grid["nidx"] = (grid["row_id"].astype(int) * n_cols + grid["column_id"].astype(int)).astype(np.int32)

    flow = np.zeros((t, n, len(data_col)), dtype=np.float32)
    for fi, col in enumerate(data_col):
        flow[grid["tidx"].to_numpy(), grid["nidx"].to_numpy(), fi] = grid[col].to_numpy(dtype=np.float32)

    adj = _grid_adjacency(n_rows, n_cols, connectivity)
    ts = pd.to_datetime(times)
    delta = int((ts[1] - ts[0]).total_seconds()) if len(ts) > 1 else 1800
    return flow, ts, adj, delta


def _grid_adjacency(n_rows: int, n_cols: int, connectivity: int = 8) -> np.ndarray:
    """Build a row-major flattened adjacency for an ``n_rows x n_cols`` grid."""
    n = n_rows * n_cols
    adj = np.zeros((n, n), dtype=np.float32)
    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        raise ValueError(connectivity)
    for r in range(n_rows):
        for c in range(n_cols):
            u = r * n_cols + c
            for dr, dc in offsets:
                rr, cc = r + dr, c + dc
                if 0 <= rr < n_rows and 0 <= cc < n_cols:
                    v = rr * n_cols + cc
                    adj[u, v] = 1.0
    return adj


def synthetic_flow(num_nodes: int = 16, num_steps: int = 1024, num_features: int = 1, seed: int = 42):
    rng = np.random.default_rng(seed)
    t = np.arange(num_steps)
    base = np.sin(2 * np.pi * t / 96.0)[:, None]
    flow = (base + rng.normal(0, 0.2, size=(num_steps, num_nodes))).astype(np.float32)
    flow = flow[..., None] if num_features == 1 else np.repeat(flow[..., None], num_features, axis=-1)
    flow = flow + 5.0  # keep positive

    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        adj[i, (i + 1) % num_nodes] = 1.0
        adj[(i + 1) % num_nodes, i] = 1.0

    timestamps = pd.date_range("2024-01-01", periods=num_steps, freq="15min")
    return flow.astype(np.float32), timestamps, adj.astype(np.float32), 900


# ---------------------------------------------------------------------------
# Sliding-window builder + scalers
# ---------------------------------------------------------------------------

@dataclass
class TrafficSplits:
    train: SlidingDataset
    val: SlidingDataset
    test: SlidingDataset
    adj: np.ndarray
    scaler: object
    num_nodes: int
    feature_dim: int


def _build_windows(
    flow: np.ndarray,
    timestamps,
    time_interval: int,
    in_window: int,
    out_window: int,
):
    n_times = flow.shape[0]
    xs, ys, tods, dows = [], [], [], []
    for i in range(n_times - in_window - out_window + 1):
        xs.append(flow[i : i + in_window])
        ys.append(flow[i + in_window : i + in_window + out_window])
        if timestamps is not None:
            ts_x = timestamps[i : i + in_window]
            tods.append(np.array([tt.hour * 60 + tt.minute for tt in ts_x], dtype=np.int64))
            dows.append(np.array([tt.dayofweek for tt in ts_x], dtype=np.int64))
    x = np.stack(xs, 0)
    y = np.stack(ys, 0)
    tod = np.stack(tods, 0) if tods else None
    dow = np.stack(dows, 0) if dows else None
    return x, y, tod, dow


def split_traffic(
    flow: np.ndarray,
    timestamps,
    time_interval: int,
    in_window: int,
    out_window: int,
    train_ratio: float,
    val_ratio: float,
    scaler_cls=None,
) -> TrafficSplits:
    """Build train/val/test ``SlidingDataset`` from a raw flow tensor.

    The scaler is fit on the train split. Compatible with ``StandardScaler``
    in ``ssl_stmformer.utils``.
    """
    from .utils import StandardScaler

    x, y, tod, dow = _build_windows(flow, timestamps, time_interval, in_window, out_window)
    n = len(x)
    n_tr = int(n * train_ratio)
    n_va = int(n * (train_ratio + val_ratio))

    scaler_cls = scaler_cls or StandardScaler
    scaler = scaler_cls().fit(x[:n_tr])
    x_norm = scaler.transform(x)
    y_norm = scaler.transform(y)

    def _slice(a, b):
        return SlidingDataset(
            x_norm[a:b], y_norm[a:b],
            time_in_day=tod[a:b] if tod is not None else None,
            day_in_week=dow[a:b] if dow is not None else None,
        )

    return TrafficSplits(
        train=_slice(0, n_tr),
        val=_slice(n_tr, n_va),
        test=_slice(n_va, n),
        adj=None,
        scaler=scaler,
        num_nodes=flow.shape[1],
        feature_dim=flow.shape[-1],
    )


def make_loaders(splits: TrafficSplits, batch_size: int, num_workers: int = 0):
    return {
        "train": DataLoader(splits.train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True),
        "val": DataLoader(splits.val, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(splits.test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
