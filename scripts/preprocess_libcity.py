"""Convert a LibCity-style ``.dyna`` dataset into pre-windowed ``.npz`` splits.

Useful when you want to cache the preprocessed tensors and reuse them across
experiments without re-parsing the raw CSV files every time.

Usage::

    python scripts/preprocess_libcity.py \
        --raw-data ./data \
        --dataset PeMS04 \
        --out data/PeMS04_npz \
        --in-window 12 --out-window 12

The output layout matches what ``source: npz`` expects, so the preprocessed
folder can be loaded back directly.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ssl_stmformer.data import load_libcity_dyna  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-data", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--in-window", type=int, default=12)
    p.add_argument("--out-window", type=int, default=12)
    p.add_argument("--train-ratio", type=float, default=0.6)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--data-col", type=str, default="traffic_flow")
    args = p.parse_args()

    flow, ts, adj, dt = load_libcity_dyna(args.raw_data, args.dataset, data_col=args.data_col)
    n_times = flow.shape[0]
    xs, ys = [], []
    for i in range(n_times - args.in_window - args.out_window + 1):
        xs.append(flow[i : i + args.in_window])
        ys.append(flow[i + args.in_window : i + args.in_window + args.out_window])
    x = np.stack(xs, 0).astype(np.float32)
    y = np.stack(ys, 0).astype(np.float32)
    n = len(x)
    n_tr = int(n * args.train_ratio)
    n_va = int(n * (args.train_ratio + args.val_ratio))

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "train.npz"), x=x[:n_tr], y=y[:n_tr])
    np.savez(os.path.join(args.out, "val.npz"), x=x[n_tr:n_va], y=y[n_tr:n_va])
    np.savez(os.path.join(args.out, "test.npz"), x=x[n_va:], y=y[n_va:])
    np.savez(os.path.join(args.out, "adj_mx.npz"), adj_mx=adj)
    print(f"saved {n} windows to {args.out}  (train {n_tr}, val {n_va - n_tr}, test {n - n_va})")
    print(f"  N={flow.shape[1]}  F={flow.shape[-1]}  interval={dt}s")


if __name__ == "__main__":
    main()
