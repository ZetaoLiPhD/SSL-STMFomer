
from __future__ import annotations

import copy
from typing import Tuple

import numpy as np
import torch


def sim_global(flow_data: torch.Tensor, sim_type: str = "cos") -> torch.Tensor:
    """Region-wise similarity from a flow tensor.

    Accepts either ``(N, L, V, C)`` flow or ``(N, V, C)`` region embeddings.
    """
    if flow_data.dim() == 4:
        n, l, v, c = flow_data.shape
        att_scaling = n * l * c
        norm = torch.norm(flow_data, p=2, dim=(0, 1, 3)) ** -1
        sim = torch.einsum("btnc, btmc->nm", flow_data, flow_data)
    elif flow_data.dim() == 3:
        n, v, c = flow_data.shape
        att_scaling = n * c
        norm = torch.norm(flow_data, p=2, dim=(0, 2)) ** -1
        sim = torch.einsum("bnc, bmc->nm", flow_data, flow_data)
    else:
        raise ValueError(f"sim_global expects rank 3 or 4, got {flow_data.dim()}")

    if sim_type == "cos":
        scaling = torch.einsum("i,j->ij", norm, norm)
        sim = sim * scaling
    elif sim_type == "att":
        sim = torch.softmax(sim * (att_scaling ** -0.5), dim=-1)
    else:
        raise ValueError(sim_type)
    return sim


def aug_topology(sim_mx: torch.Tensor, input_graph: torch.Tensor, percent: float = 0.2) -> torch.Tensor:
    """Drop low-similarity edges and add high-similarity edges to the graph."""
    drop_percent = percent / 2.0
    index_list = input_graph.nonzero()
    edge_num = max(1, int(index_list.shape[0] / 2))
    edge_mask = (input_graph > 0).tril(diagonal=-1)
    add_drop_num = max(1, int(edge_num * drop_percent / 2))
    aug_graph = copy.deepcopy(input_graph)

    if edge_mask.sum() > 0:
        drop_prob = torch.softmax(sim_mx[edge_mask], dim=0).cpu()
        drop_prob = (1.0 - drop_prob).numpy()
        if drop_prob.sum() > 0:
            drop_prob /= drop_prob.sum()
            drop_list = np.random.choice(edge_num, size=add_drop_num, p=drop_prob)
            drop_index = index_list[drop_list]
            zeros = torch.zeros_like(aug_graph[0, 0])
            aug_graph[drop_index[:, 0], drop_index[:, 1]] = zeros
            aug_graph[drop_index[:, 1], drop_index[:, 0]] = zeros

    node_num = input_graph.shape[0]
    x, y = np.meshgrid(range(node_num), range(node_num), indexing="ij")
    mask = y < x
    x, y = x[mask], y[mask]
    add_prob = sim_mx[torch.ones(sim_mx.size(), dtype=torch.bool).tril(diagonal=-1)]
    add_prob = torch.softmax(add_prob, dim=0).cpu().numpy()
    if add_prob.sum() > 0:
        add_list = np.random.choice(
            int((node_num * node_num - node_num) / 2),
            size=add_drop_num,
            p=add_prob,
        )
        ones = torch.ones_like(aug_graph[0, 0])
        aug_graph[x[add_list], y[add_list]] = ones
        aug_graph[y[add_list], x[add_list]] = ones
    return aug_graph


def aug_traffic(t_sim_mx: torch.Tensor, flow_data: torch.Tensor, percent: float = 0.2) -> torch.Tensor:
    """Mask the flow values whose temporal similarity score is the lowest."""
    l, n, v = t_sim_mx.shape
    mask_num = max(1, int(n * l * v * percent))
    aug_flow = copy.deepcopy(flow_data)
    raw = (1.0 - t_sim_mx.permute(1, 0, 2).reshape(-1)).cpu().numpy()
    mask_prob = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    mask_prob = np.clip(mask_prob, a_min=0.0, a_max=None)
    total = mask_prob.sum()
    if not np.isfinite(total) or total <= 0:
        # Fall back to a uniform random mask -- the SSL signal is weak, but
        # at least training does not crash.
        mask_prob = np.ones_like(mask_prob) / mask_prob.size
    else:
        mask_prob = mask_prob / total
    x, y, z = np.meshgrid(range(n), range(l), range(v), indexing="ij")
    mask_list = np.random.choice(n * l * v, size=mask_num, p=mask_prob)
    zeros = torch.zeros_like(aug_flow[0, 0, 0])
    aug_flow[
        x.reshape(-1)[mask_list],
        y.reshape(-1)[mask_list],
        z.reshape(-1)[mask_list],
    ] = zeros
    return aug_flow


def aug_traffic_gpu(t_sim_mx: torch.Tensor, flow_data: torch.Tensor, percent: float = 0.1) -> torch.Tensor:
    """GPU-resident version of ``aug_traffic`` for use in the training loop.

    ``t_sim_mx``: ``(T, B, N)`` softmaxed similarity, values in ``[0, 1]``.
    ``flow_data``: ``(B, T, N, C)`` input flow.

    For every cell we draw a Bernoulli mask with probability proportional to
    ``1 - t_sim_mx`` (so less-similar cells are more likely to be dropped),
    rescaled so that on average ``percent`` of cells are masked.
    """
    if percent <= 0:
        return flow_data
    sim = torch.nan_to_num(t_sim_mx, nan=0.0, posinf=1.0, neginf=0.0)
    sim = sim.permute(1, 0, 2)              # (B, T, N)
    weight = (1.0 - sim).clamp_min(0.0)
    # Normalise per-sample so expected mask fraction == ``percent``.
    sample_mean = weight.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    drop_prob = (weight / sample_mean * percent).clamp_max(1.0)
    keep = (torch.rand_like(drop_prob) > drop_prob).float().unsqueeze(-1)
    return flow_data * keep
