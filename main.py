"""Top-level entry point for SSL-STMFormer training and evaluation.

Examples
--------

    # 1-epoch synthetic smoke test on CPU
    python main.py --config configs/synthetic.yaml --epochs 1 --device cpu

    # Train on PeMS04 (LibCity .dyna raw data)
    python main.py --config configs/PeMS04.yaml

    # Train on NYCBike1 (pre-windowed .npz splits)
    python main.py --config configs/NYCBike1.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Allow running both as module and script.
_PKG_PARENT = os.path.dirname(os.path.abspath(__file__))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from ssl_stmformer.data import (
    SlidingDataset,
    load_libcity_dyna,
    load_libcity_grid,
    load_npz,
    make_loaders,
    split_traffic,
    synthetic_flow,
)
from ssl_stmformer.model import SSLSTMFormer, SSLSTMFormerConfig
from ssl_stmformer.trainer import Trainer, TrainConfig
from ssl_stmformer.utils import (
    StandardScaler,
    count_parameters,
    dtw_matrix_cached,
    dtw_topk_mask,
    hop_distance_cached,
    laplacian_eigenvectors,
    set_seed,
)


def _load_config(path: str):
    p = Path(path).resolve()
    with open(p, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg, p.parent


def _resolve(cfg_path_dir: Path, p: str) -> str:
    rel = Path(p)
    if rel.is_absolute():
        return str(rel)
    return str((cfg_path_dir / rel).resolve())


def _build_dataset(cfg, cfg_dir: Path):
    """Return a TrafficSplits-like object plus adjacency matrix."""
    data_cfg = cfg["dataset"]
    src = data_cfg["source"]

    if src == "synthetic":
        flow, ts, adj, dt = synthetic_flow(
            num_nodes=data_cfg.get("num_nodes", 16),
            num_steps=data_cfg.get("num_steps", 1024),
            num_features=data_cfg.get("feature_dim", 1),
        )
        splits = split_traffic(
            flow, ts, dt,
            in_window=data_cfg["input_window"],
            out_window=data_cfg["output_window"],
            train_ratio=data_cfg.get("train_ratio", 0.6),
            val_ratio=data_cfg.get("val_ratio", 0.2),
        )
        splits.adj = adj
        splits.raw_flow = flow
        splits.time_interval = dt
        return splits

    if src == "npz":
        root = _resolve(cfg_dir, data_cfg["root"])
        data, adj = load_npz(root, data_cfg["name"])
        # Splits are already pre-windowed; we just need to fit the scaler.
        scaler = StandardScaler().fit(np.concatenate([data["train"]["x"], data["val"]["x"]], axis=0))
        train_x = scaler.transform(data["train"]["x"])
        train_y = scaler.transform(data["train"]["y"])
        val_x = scaler.transform(data["val"]["x"])
        val_y = scaler.transform(data["val"]["y"])
        test_x = scaler.transform(data["test"]["x"])
        test_y = scaler.transform(data["test"]["y"])

        from ssl_stmformer.data import TrafficSplits

        splits = TrafficSplits(
            train=SlidingDataset(train_x, train_y),
            val=SlidingDataset(val_x, val_y),
            test=SlidingDataset(test_x, test_y),
            adj=adj,
            scaler=scaler,
            num_nodes=adj.shape[0],
            feature_dim=train_x.shape[-1],
        )
        splits.raw_flow = None
        splits.time_interval = int(data_cfg.get("time_interval", 1800))
        return splits

    if src == "libcity_dyna":
        root = _resolve(cfg_dir, data_cfg["root"])
        flow, ts, adj, dt = load_libcity_dyna(
            root, data_cfg["name"], data_col=data_cfg.get("data_col", ("traffic_flow",))
        )
        splits = split_traffic(
            flow, ts, dt,
            in_window=data_cfg["input_window"],
            out_window=data_cfg["output_window"],
            train_ratio=data_cfg.get("train_ratio", 0.6),
            val_ratio=data_cfg.get("val_ratio", 0.2),
        )
        splits.adj = adj
        splits.raw_flow = flow
        splits.time_interval = dt
        return splits

    if src == "libcity_grid":
        root = _resolve(cfg_dir, data_cfg["root"])
        flow, ts, adj, dt = load_libcity_grid(
            root, data_cfg["name"],
            data_col=data_cfg.get("data_col", ("inflow", "outflow")),
            connectivity=data_cfg.get("grid_connectivity", 8),
        )
        splits = split_traffic(
            flow, ts, dt,
            in_window=data_cfg["input_window"],
            out_window=data_cfg["output_window"],
            train_ratio=data_cfg.get("train_ratio", 0.7),
            val_ratio=data_cfg.get("val_ratio", 0.1),
        )
        splits.adj = adj
        splits.raw_flow = flow
        splits.time_interval = dt
        return splits

    raise ValueError(f"Unknown dataset source: {src}")


def _build_model(cfg, splits):
    m_cfg = cfg["model"]
    config = SSLSTMFormerConfig(
        num_nodes=splits.num_nodes,
        feature_dim=splits.feature_dim,
        output_dim=m_cfg.get("output_dim", splits.feature_dim),
        embed_dim=m_cfg.get("embed_dim", 64),
        skip_dim=m_cfg.get("skip_dim", 256),
        lape_dim=m_cfg.get("lape_dim", 8),
        input_window=cfg["dataset"]["input_window"],
        output_window=cfg["dataset"]["output_window"],
        enc_depth=m_cfg.get("enc_depth", 4),
        head_dim=m_cfg.get("head_dim", 8),
        spatial_heads_per_kind=m_cfg.get("spatial_heads_per_kind", 2),
        temporal_heads_per_kind=m_cfg.get("temporal_heads_per_kind", 2),
        mlp_ratio=m_cfg.get("mlp_ratio", 4.0),
        drop=m_cfg.get("drop", 0.1),
        attn_drop=m_cfg.get("attn_drop", 0.0),
        mask_ratio=m_cfg.get("mask_ratio", 0.1),
        short_lambda=m_cfg.get("short_lambda", 3),
        long_topk=m_cfg.get("long_topk", 5),
        add_time_in_day=m_cfg.get("add_time_in_day", True),
        add_day_in_week=m_cfg.get("add_day_in_week", True),
        n_prototypes=m_cfg.get("n_prototypes", 16),
        shm_temp=m_cfg.get("shm_temp", 0.5),
        aug_percent=m_cfg.get("aug_percent", 0.1),
        use_ssl=m_cfg.get("use_ssl", True),
    )
    return SSLSTMFormer(config), config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint for --mode test")
    args = parser.parse_args()

    cfg, cfg_dir = _load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["lr"] = args.lr
    if args.device is not None:
        cfg["train"]["device"] = args.device

    set_seed(args.seed)

    splits = _build_dataset(cfg, cfg_dir)
    print(
        f"[data] source={cfg['dataset']['source']}  N={splits.num_nodes}  F={splits.feature_dim}  "
        f"train/val/test={len(splits.train)}/{len(splits.val)}/{len(splits.test)}"
    )

    model, m_cfg = _build_model(cfg, splits)

    # --- precompute static masks (cached on disk) ---------------------
    adj = splits.adj
    cache_root = _resolve(cfg_dir, cfg.get("cache_dir", "cache"))
    os.makedirs(cache_root, exist_ok=True)
    name = cfg["dataset"].get("name", cfg["dataset"]["source"])
    print(f"[setup] computing Laplacian PE / hop / DTW (cache={cache_root}/{name}) ...")

    lap = laplacian_eigenvectors(adj, m_cfg.lape_dim)
    hop = hop_distance_cached(adj, cache_path=os.path.join(cache_root, f"{name}_hop.npy"))

    dtw_mask = None
    if "long" in m_cfg.use_spatial_strategies:
        if splits.raw_flow is None:
            print("[setup] no raw flow available; skipping DTW long-range mask")
        else:
            points_per_hour = max(1, 3600 // max(1, splits.time_interval))
            dtw_path = os.path.join(cache_root, f"{name}_dtw.npy")
            dtw_dist = dtw_matrix_cached(splits.raw_flow, points_per_hour=points_per_hour, cache_path=dtw_path)
            dtw_mask = dtw_topk_mask(dtw_dist, k=m_cfg.long_topk)

    model.set_static_masks(hop, dtw_mask=dtw_mask)
    print(f"[model] params={count_parameters(model)/1e6:.2f}M")

    # --- assemble loaders ---------------------------------------------
    loaders = make_loaders(splits, batch_size=cfg["train"]["batch_size"], num_workers=cfg["train"].get("num_workers", 0))

    # --- training -----------------------------------------------------
    train_cfg = TrainConfig(
        epochs=cfg["train"]["epochs"],
        batch_size=cfg["train"]["batch_size"],
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
        grad_clip=float(cfg["train"].get("grad_clip", 5.0)),
        early_stop_patience=int(cfg["train"].get("early_stop_patience", 15)),
        use_dwa=cfg["train"].get("use_dwa", True),
        dwa_temp=float(cfg["train"].get("dwa_temp", 4.0)),
        log_every=int(cfg["train"].get("log_every", 50)),
        ckpt_dir=_resolve(cfg_dir, cfg["train"].get("ckpt_dir", "checkpoints")),
        device=cfg["train"].get("device", "cuda"),
        mask_value=float(cfg["train"].get("mask_value", 0.0)),
    )
    device = torch.device(train_cfg.device if torch.cuda.is_available() or train_cfg.device == "cpu" else "cpu")
    model.to(device)
    lap_t = torch.from_numpy(lap).float().to(device)
    graph_t = torch.from_numpy(adj).float().to(device)

    trainer = Trainer(model, loaders, splits.scaler, graph_t, lap_t, train_cfg)

    if args.mode == "train":
        trainer.fit()
    else:
        ckpt_path = args.ckpt or trainer.best_path
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        metrics, _, _ = trainer.test()
        print("Test metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
