

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .utils import dwa, ensure_dir, masked_mae, masked_mape, masked_rmse, metric_dict


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    early_stop_patience: int = 15
    use_dwa: bool = True
    dwa_temp: float = 4.0
    log_every: int = 50
    ckpt_dir: str = "checkpoints"
    mask_value: float = 0.0
    device: str = "cuda"


class Trainer:


    def __init__(
        self,
        model,
        loaders,
        scaler,
        graph,
        lap_mx,
        train_cfg: TrainConfig,
    ):
        self.model = model
        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]
        self.test_loader = loaders["test"]
        self.scaler = scaler
        self.graph = graph
        self.lap_mx = lap_mx
        self.cfg = train_cfg

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
        )
        self.device = torch.device(train_cfg.device if torch.cuda.is_available() or train_cfg.device == "cpu" else "cpu")

        ensure_dir(train_cfg.ckpt_dir)
        self.best_path = os.path.join(train_cfg.ckpt_dir, "best.pt")

    # ------------------------------------------------------------------

    def _move(self, batch):
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    def train_epoch(self, loss_weights):
        self.model.train()
        total_loss = 0.0
        sep = np.zeros(3, dtype=np.float64)
        for step, batch in enumerate(self.train_loader):
            batch = self._move(batch)
            x = batch["X"]
            y = batch["y"]
            tod = batch.get("time_in_day")
            dow = batch.get("day_in_week")

            repr1, repr2, pred = self.model(
                x, self.graph, self.lap_mx, time_in_day=tod, day_in_week=dow,
            )
            l_p, l_s, l_t = self.model.compute_losses(repr1, repr2, pred, y)
            loss = loss_weights[0] * l_p + loss_weights[1] * l_s + loss_weights[2] * l_t

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            sep += np.array([l_p.item(), l_s.item(), l_t.item()])

            if (step + 1) % self.cfg.log_every == 0:
                print(
                    f"  step {step+1:>4}/{len(self.train_loader)}  "
                    f"loss={loss.item():.4f}  "
                    f"l_p={l_p.item():.4f}  l_s={l_s.item():.4f}  l_t={l_t.item():.4f}"
                )

        n = max(1, len(self.train_loader))
        return total_loss / n, sep / n

    @torch.no_grad()
    def evaluate(self, loader, loss_weights):
        self.model.eval()
        total = 0.0
        for batch in loader:
            batch = self._move(batch)
            x = batch["X"]
            y = batch["y"]
            tod = batch.get("time_in_day")
            dow = batch.get("day_in_week")
            repr1, repr2, pred = self.model(
                x, self.graph, self.lap_mx, time_in_day=tod, day_in_week=dow,
                augment=False,
            )
            l_p, l_s, l_t = self.model.compute_losses(repr1, repr2, pred, y)
            loss = loss_weights[0] * l_p + loss_weights[1] * l_s + loss_weights[2] * l_t
            total += loss.item()
        return total / max(1, len(loader))

    @torch.no_grad()
    def test(self, loader=None):
        loader = loader or self.test_loader
        self.model.eval()
        preds, trues = [], []
        for batch in loader:
            batch = self._move(batch)
            x = batch["X"]
            y = batch["y"]
            tod = batch.get("time_in_day")
            dow = batch.get("day_in_week")
            _, _, pred = self.model(
                x, self.graph, self.lap_mx, time_in_day=tod, day_in_week=dow,
                augment=False,
            )
            preds.append(pred.cpu())
            trues.append(y.cpu())
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        # Inverse-scale before computing metrics on the original scale.
        if self.scaler is not None:
            preds = torch.tensor(self.scaler.inverse_transform(preds.numpy()))
            trues = torch.tensor(self.scaler.inverse_transform(trues.numpy()))
        out = metric_dict(preds, trues, mask_value=self.cfg.mask_value)
        return out, preds, trues

    # ------------------------------------------------------------------

    def fit(self):
        best_val = float("inf")
        best_epoch = 0
        not_improved = 0
        loss_tm1 = np.ones(3)
        loss_tm2 = np.ones(3)
        weights = np.ones(3)

        start = time.time()
        for epoch in range(1, self.cfg.epochs + 1):
            if self.cfg.use_dwa:
                if epoch <= 2:
                    weights = np.ones(3)
                else:
                    weights = dwa(loss_tm1, loss_tm2, T=self.cfg.dwa_temp)

            train_loss, sep = self.train_epoch(weights)
            loss_tm2 = loss_tm1
            loss_tm1 = sep

            val_loss = self.evaluate(self.val_loader, weights)
            print(
                f"Epoch {epoch:03d}/{self.cfg.epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"weights={weights.round(3).tolist()}"
            )

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                not_improved = 0
                torch.save(
                    {"model": self.model.state_dict(), "epoch": epoch, "best_val": best_val},
                    self.best_path,
                )
                print(f"  -> saved best model to {self.best_path}")
            else:
                not_improved += 1
                if not_improved >= self.cfg.early_stop_patience:
                    print(f"Early stop at epoch {epoch} (no improvement for {not_improved} epochs)")
                    break

        elapsed = (time.time() - start) / 60.0
        print(f"== Training done in {elapsed:.2f} min  best_val={best_val:.4f}@{best_epoch}")

        # Reload best and evaluate on test split.
        ckpt = torch.load(self.best_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        metrics, _, _ = self.test()
        print("== Test ==")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        return {"best_val": best_val, "best_epoch": best_epoch, "test": metrics}
