

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .augment import aug_traffic_gpu, sim_global
from .embedding import DataEmbedding
from .encoder import STAttentionConfig, STEncoderLayer
from .masks import long_range_mask, short_range_mask
from .ssl import SpatialHeteroModel, TemporalHeteroModel


@dataclass
class SSLSTMFormerConfig:
    num_nodes: int
    feature_dim: int = 1
    output_dim: int = 1

    embed_dim: int = 64
    skip_dim: int = 256
    lape_dim: int = 8

    input_window: int = 12
    output_window: int = 12

    enc_depth: int = 4
    head_dim: int = 8
    spatial_heads_per_kind: int = 2
    temporal_heads_per_kind: int = 2
    mlp_ratio: float = 4.0
    drop: float = 0.1
    attn_drop: float = 0.0
    mask_ratio: float = 0.1

    short_lambda: int = 3                 # hop threshold for M_short
    long_topk: int = 5                    # K for M_long (top-K DTW neighbours)

    add_time_in_day: bool = True
    add_day_in_week: bool = True

    use_spatial_strategies: Sequence[str] = field(
        default_factory=lambda: ("rand", "tube", "block", "short", "long")
    )
    use_temporal_strategies: Sequence[str] = field(
        default_factory=lambda: ("rand", "tube", "block", "temp", "plain")
    )

    # SSL heads
    n_prototypes: int = 16
    shm_temp: float = 0.5
    aug_percent: float = 0.1
    use_ssl: bool = True


class _OutputHead(nn.Module):
    """1x1 -> 1x1 convolutional head used after the encoder stack.

    Implements the output projection described in the paper: per-layer
    1x1 convs reduce ``embed_dim`` to ``skip_dim``, the layer outputs are
    summed, and two more 1x1 convs map ``T -> T'`` and ``skip_dim -> output_dim``.
    """

    def __init__(self, embed_dim: int, skip_dim: int, input_window: int, output_window: int, output_dim: int):
        super().__init__()
        self.skip_dim = skip_dim
        self.skip_convs = nn.ModuleList()  # populated by caller
        self.end_conv1 = nn.Conv2d(input_window, output_window, kernel_size=1, bias=True)
        self.end_conv2 = nn.Conv2d(skip_dim, output_dim, kernel_size=1, bias=True)


class SSLSTMFormer(nn.Module):
    def __init__(self, cfg: SSLSTMFormerConfig):
        super().__init__()
        self.cfg = cfg

        # --- embedding ---------------------------------------------------
        self.embedding = DataEmbedding(
            feature_dim=cfg.feature_dim,
            embed_dim=cfg.embed_dim,
            lape_dim=cfg.lape_dim,
            add_time_in_day=cfg.add_time_in_day,
            add_day_in_week=cfg.add_day_in_week,
            dropout=cfg.drop,
        )

        # --- encoder -----------------------------------------------------
        attn_cfg = STAttentionConfig(
            embed_dim=cfg.embed_dim,
            head_dim=cfg.head_dim,
            spatial_heads_per_kind=cfg.spatial_heads_per_kind,
            temporal_heads_per_kind=cfg.temporal_heads_per_kind,
            attn_dropout=cfg.attn_drop,
            proj_dropout=cfg.drop,
            mask_ratio=cfg.mask_ratio,
            use_spatial_strategies=cfg.use_spatial_strategies,
            use_temporal_strategies=cfg.use_temporal_strategies,
        )
        self.encoder_blocks = nn.ModuleList([
            STEncoderLayer(attn_cfg, mlp_ratio=cfg.mlp_ratio, dropout=cfg.drop)
            for _ in range(cfg.enc_depth)
        ])

        # --- output projection ------------------------------------------
        self.skip_convs = nn.ModuleList([
            nn.Conv2d(cfg.embed_dim, cfg.skip_dim, kernel_size=1) for _ in range(cfg.enc_depth)
        ])
        self.end_conv1 = nn.Conv2d(cfg.input_window, cfg.output_window, kernel_size=1, bias=True)
        self.end_conv2 = nn.Conv2d(cfg.skip_dim, cfg.output_dim, kernel_size=1, bias=True)

        # --- SSL heads ---------------------------------------------------
        if cfg.use_ssl:
            self.shm = SpatialHeteroModel(cfg.skip_dim, cfg.n_prototypes, cfg.shm_temp)
            self.thm = TemporalHeteroModel(cfg.skip_dim, cfg.num_nodes)

        # Static masks are filled by ``set_static_masks`` after construction.
        self.register_buffer("short_mask", torch.empty(0))
        self.register_buffer("long_mask", torch.empty(0))

    # ------------------------------------------------------------------
    # Static mask setup
    # ------------------------------------------------------------------

    def set_static_masks(self, hop_mx, dtw_mask=None) -> None:
        """Provide the precomputed hop / DTW matrices.

        ``hop_mx``  : ``(N, N)`` integer matrix of shortest-path hops.
        ``dtw_mask``: ``(N, N)`` 0/1 mask of top-K DTW neighbours. If ``None``
                      the long-range strategy falls back to identity (only
                      self-attention is allowed for the long-range head).
        """
        sm = short_range_mask(hop_mx, self.cfg.short_lambda)
        self.short_mask = torch.as_tensor(sm, dtype=torch.float32)
        if dtw_mask is None:
            dtw_mask = torch.eye(self.cfg.num_nodes).numpy()
        lm = long_range_mask(dtw_mask)
        self.long_mask = torch.as_tensor(lm, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(
        self,
        x: torch.Tensor,
        lap_mx: torch.Tensor,
        time_in_day: Optional[torch.Tensor] = None,
        day_in_week: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run embedding + encoder + skip aggregation.

        Returns ``(B, T, N, skip_dim)`` hidden states.
        """
        h = self.embedding(x, lap_mx, time_in_day, day_in_week)
        skip = 0
        for i, block in enumerate(self.encoder_blocks):
            h = block(h, self.short_mask.to(h.device), self.long_mask.to(h.device))
            # (B, T, N, D) -> (B, D, N, T) for Conv2d, then back.
            sk = self.skip_convs[i](h.permute(0, 3, 2, 1))
            skip = sk if not torch.is_tensor(skip) else skip + sk
        # Permute back to (B, T, N, skip_dim).
        return skip.permute(0, 3, 2, 1)

    def predict(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map ``(B, T, N, skip_dim)`` hidden states to ``(B, T', N, output_dim)``."""
        # end_conv1 mixes along the time axis (T -> T').
        # We must put time as the channel axis: (B, T, N, D) -> (B, T, D, N).
        x = F.relu(hidden.permute(0, 1, 3, 2))
        x = self.end_conv1(x)                              # (B, T', D, N)
        x = F.relu(x.permute(0, 2, 1, 3))                  # (B, D, T', N) - move D to channel
        x = self.end_conv2(x)                              # (B, output_dim, T', N)
        return x.permute(0, 2, 3, 1)                       # (B, T', N, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        graph: torch.Tensor,
        lap_mx: torch.Tensor,
        time_in_day: Optional[torch.Tensor] = None,
        day_in_week: Optional[torch.Tensor] = None,
        augment: bool = True,
    ):
        """Two-view forward pass.

        Returns ``(repr1, repr2, pred)`` where ``repr1`` / ``repr2`` are the
        skip-aggregated hidden states for the two views and ``pred`` is the
        forecast computed from ``repr1``.
        """
        # --- view 1 -----------------------------------------------------
        repr1 = self.encode(x, lap_mx, time_in_day, day_in_week)
        pred = self.predict(repr1)

        if not augment or not self.training:
            return repr1, repr1, pred

        # --- view 2: adaptive flow augmentation (all on GPU) -----------
        # We keep the augmentation on-device to avoid the per-batch
        # GPU<->CPU sync that would otherwise dominate runtime. The
        # Laplacian PE is computed once on the original adjacency rather
        # than every batch on the augmented graph; the SSL signal is
        # carried by the flow-level augmentation only.
        with torch.no_grad():
            t_sim_mx = self._temporal_sim(repr1)             # (T, B, N)
            x_aug = aug_traffic_gpu(t_sim_mx, x, percent=self.cfg.aug_percent)
        repr2 = self.encode(x_aug, lap_mx, time_in_day, day_in_week)
        return repr1, repr2, pred

    @staticmethod
    def _temporal_sim(repr1: torch.Tensor) -> torch.Tensor:
        """Build the temporal similarity tensor used by ``aug_traffic``.

        Each ``(b, t, n)`` embedding is compared against the node's
        time-mean via cosine similarity, then softmax-normalised over the
        time axis so every ``(b, n)`` position sums to 1. This gives a
        per-step importance distribution suitable for adaptive masking.
        """
        b, t, n, c = repr1.shape
        norm = F.normalize(repr1, dim=-1, eps=1e-6)
        mean_t = norm.mean(dim=1, keepdim=True)               # (B, 1, N, C)
        sim = (norm * mean_t).sum(dim=-1)                     # (B, T, N)
        sim = torch.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
        sim = sim.permute(1, 0, 2)                            # (T, B, N)
        return torch.softmax(sim, dim=0)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_losses(
        self,
        repr1: torch.Tensor,
        repr2: torch.Tensor,
        pred: torch.Tensor,
        target: torch.Tensor,
    ):
        """Returns ``(L_p, L_s, L_t)``.

        - ``L_p``: prediction MAE between ``pred`` and ``target``.
        - ``L_s``: spatial-heterogeneity contrastive cluster loss.
        - ``L_t``: temporal-heterogeneity contrastive loss.
        """
        l_p = F.l1_loss(pred, target[..., : self.cfg.output_dim])

        if not self.cfg.use_ssl:
            zero = torch.zeros((), device=pred.device)
            return l_p, zero, zero

        l_s = self.shm(repr1, repr2)
        l_t = self.thm(repr1, repr2)
        return l_p, l_s, l_t
