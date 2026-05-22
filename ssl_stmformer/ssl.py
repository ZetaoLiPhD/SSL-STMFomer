
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Spatial Heterogeneity Modelling
# ---------------------------------------------------------------------------

@torch.no_grad()
def _sinkhorn(out: torch.Tensor, epsilon: float = 0.05, n_iter: int = 3) -> torch.Tensor:
    """Sinkhorn-Knopp normalisation used in SwAV-style soft clustering."""
    Q = torch.exp(out / epsilon).t()  # (K, B)
    Q = Q / Q.sum()
    K, B = Q.shape
    for _ in range(n_iter):
        Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-8)
        Q = Q / K
        Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-8)
        Q = Q / B
    Q = Q * B
    return Q.t()


class SpatialHeteroModel(nn.Module):
    """Eq. (12)-(13) of the paper: prototype-based contrastive clustering."""

    def __init__(self, c_in: int, n_prototypes: int, tau: float = 0.5):
        super().__init__()
        self.prototypes = nn.Linear(c_in, n_prototypes, bias=False)
        self.tau = tau
        self.c_in = c_in
        nn.init.xavier_uniform_(self.prototypes.weight)

    @staticmethod
    def _l2(x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1, p=2)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """``z1, z2``: ``(B, T, N, C)`` representations of two augmented views."""
        with torch.no_grad():
            self.prototypes.weight.data = self._l2(self.prototypes.weight.data)
        zc1 = self.prototypes(self._l2(z1.reshape(-1, self.c_in)))
        zc2 = self.prototypes(self._l2(z2.reshape(-1, self.c_in)))
        with torch.no_grad():
            q1 = _sinkhorn(zc1.detach())
            q2 = _sinkhorn(zc2.detach())
        l1 = -torch.mean(torch.sum(q1 * F.log_softmax(zc2 / self.tau, dim=1), dim=1))
        l2 = -torch.mean(torch.sum(q2 * F.log_softmax(zc1 / self.tau, dim=1), dim=1))
        return l1 + l2


# ---------------------------------------------------------------------------
# Temporal Heterogeneity Modelling
# ---------------------------------------------------------------------------

class _AvgReadout(nn.Module):
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, N, C) -> (B, C)
        return torch.sigmoid(torch.mean(h, dim=1))


class _Discriminator(nn.Module):
    def __init__(self, n_h: int):
        super().__init__()
        self.net = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            if isinstance(m, nn.Bilinear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, summary: torch.Tensor, h_real: torch.Tensor, h_fake: torch.Tensor) -> torch.Tensor:
        s = summary.unsqueeze(1).expand_as(h_real).contiguous()
        sc_real = torch.squeeze(self.net(h_real, s), dim=2)
        sc_fake = torch.squeeze(self.net(h_fake, s), dim=2)
        return torch.cat((sc_real, sc_fake), dim=1)


class TemporalHeteroModel(nn.Module):


    def __init__(self, c_in: int, num_nodes: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(num_nodes, c_in))
        self.W2 = nn.Parameter(torch.empty(num_nodes, c_in))
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        self.read = _AvgReadout()
        self.disc = _Discriminator(c_in)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """``z1, z2``: ``(B, T, N, C)``. The contrastive task is computed at
        every time step and averaged over the time axis.
        """
        b, t, n, c = z1.shape
        loss = 0.0
        for ti in range(t):
            h = z1[:, ti] * self.W1 + z2[:, ti] * self.W2  # (B, N, C)
            s = self.read(h)
            idx = torch.randperm(b, device=z1.device)
            shuf_h = h[idx]
            logits = self.disc(s, h, shuf_h)
            lbl_real = torch.ones(b, n, device=z1.device)
            lbl_fake = torch.zeros(b, n, device=z1.device)
            lbl = torch.cat((lbl_real, lbl_fake), dim=1)
            loss = loss + self.bce(logits, lbl)
        return loss / max(t, 1)
