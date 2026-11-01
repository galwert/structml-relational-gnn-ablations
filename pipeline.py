"""Shared helpers for the RelBench-reference pipeline experiments.

All A1/A2/A3/A4 rel-hm reruns import from here. Key pieces:

* :func:`build_graph_data` — cached ``make_pkey_fkey_graph`` build with a
  sentence-transformer text embedder. Wall time: ~7 min the first call for
  each dataset, ~10 s on subsequent calls (cache hit).
* :func:`build_loaders` — RelBench's ``get_node_train_table_input`` +
  :class:`NeighborLoader` with the paper's hyperparameters.
* :func:`safe_hetero_forward` — thin wrapper around
  :meth:`HeteroGraphSAGE.forward` that (a) filters out edge types with 0 edges
  in the sampled subgraph and (b) carries the previous representation forward
  for node types that receive no incoming messages. Without it the D-direction
  ablation crashes on isolated node types.
* :func:`compute_metrics` + :func:`find_best_f1` — the paper's four metrics
  plus a threshold sweep so F1 isn't stuck at 0 or ~0.90 on imbalanced tasks.
"""
from __future__ import annotations
import os, time, math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.modeling.graph import make_pkey_fkey_graph, get_node_train_table_input
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder
from relbench.modeling.utils import get_stype_proposal
from torch_frame.config.text_embedder import TextEmbedderConfig

from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)

# ---------------------------------------------------------------------------
# Shared paper hyperparameters (from RelBench appendix + a1_directionality's original)
# ---------------------------------------------------------------------------
CHANNELS = 128
NUM_LAYERS = 2
LR = 5e-3
BATCH_SIZE = 512
NUM_NEIGHBORS = 128
EPOCHS = 10
AGGR = "sum"
TEMPORAL_STRATEGY = "uniform"
MAX_STEPS_PER_EPOCH = 2000
SEED = 42


# ---------------------------------------------------------------------------
# Text embedder — small sentence-transformer, NaN-tolerant. Paper uses GloVe,
# but the interface HeteroEncoder needs is just (List[str] -> Tensor).
# ---------------------------------------------------------------------------
class SentenceEmbedder:
    def __init__(self, device: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=device)
        self._dim = 384

    def __call__(self, sentences):
        clean = []
        for s in sentences:
            if s is None:
                clean.append("")
            elif isinstance(s, float) and math.isnan(s):
                clean.append("")
            else:
                clean.append(str(s))
        with torch.no_grad():
            emb = self.model.encode(
                clean, batch_size=256, show_progress_bar=False,
                convert_to_numpy=False, convert_to_tensor=True)
        return emb

    @property
    def out_dim(self):
        return self._dim


# ---------------------------------------------------------------------------
# Cached data build. Sharing the cache_dir across A1..A4 means only the first
# script pays the ~7 min build cost; the rest hit disk cache.
# ---------------------------------------------------------------------------
def build_graph_data(dataset_name: str, task_name: str, device: str, hb=print):
    hb(f"  loading {dataset_name}/{task_name}")
    ds = get_dataset(dataset_name, download=False)
    task = get_task(dataset_name, task_name, download=False)
    db = ds.get_db()

    hb("  building stype proposal + text embedder")
    col_to_stype_dict = get_stype_proposal(db)
    embedder = SentenceEmbedder(device=device)
    text_cfg = TextEmbedderConfig(text_embedder=embedder, batch_size=256)

    hb("  make_pkey_fkey_graph ...")
    cache_dir = os.path.join(
        "/home/wertheizer/structeredML/project/results/graph_cache", dataset_name)
    data, col_stats_dict = make_pkey_fkey_graph(
        db, col_to_stype_dict,
        text_embedder_cfg=text_cfg,
        cache_dir=cache_dir,
    )
    return data, col_stats_dict, task, col_to_stype_dict


def build_loaders(data: HeteroData, task, batch_size: int = BATCH_SIZE,
                  num_neighbors: int = NUM_NEIGHBORS,
                  num_layers: int = NUM_LAYERS,
                  temporal_strategy: str = TEMPORAL_STRATEGY):
    train_table_input = get_node_train_table_input(
        table=task.get_table("train"), task=task)
    val_table_input = get_node_train_table_input(
        table=task.get_table("val"), task=task)
    neighbors = [int(num_neighbors / 2**i) for i in range(num_layers)]
    train_loader = NeighborLoader(
        data, num_neighbors=neighbors,
        time_attr="time", input_nodes=train_table_input.nodes,
        input_time=train_table_input.time,
        transform=train_table_input.transform,
        batch_size=batch_size, temporal_strategy=temporal_strategy,
        shuffle=True, num_workers=0,
    )
    val_loader = NeighborLoader(
        data, num_neighbors=neighbors,
        time_attr="time", input_nodes=val_table_input.nodes,
        input_time=val_table_input.time,
        transform=val_table_input.transform,
        batch_size=batch_size, temporal_strategy=temporal_strategy,
        shuffle=False, num_workers=0,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Safe forward for HeteroGraphSAGE.
#
# Two robustness fixes on top of :meth:`HeteroGraphSAGE.forward`:
#
# 1. **Empty-edge filter** — a sampled sub-batch can produce an edge type with
#    zero edges. SAGEConv then aggregates a ``None`` message → crash inside
#    ``aggr/base.py`` at ``x.dim()``. We drop those edge types from the dict
#    before the ``HeteroConv`` call.
# 2. **Carry-forward** — for the "D" (FK-only) ablation we drop every ``rev_``
#    edge type. Any node that only received messages via a reverse edge now
#    receives none → its key disappears from ``HeteroConv``'s output → next
#    layer's messages-from-it become undefined. We reinstate its previous
#    representation before the LayerNorm+ReLU.
# ---------------------------------------------------------------------------
def safe_hetero_forward(gnn: HeteroGraphSAGE,
                         x_dict: Dict[str, torch.Tensor],
                         edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                         ) -> Dict[str, torch.Tensor]:
    ei_filtered = {et: ei for et, ei in edge_index_dict.items() if ei.numel() > 0}
    for conv, norm_dict in zip(gnn.convs, gnn.norms):
        out = conv(x_dict, ei_filtered)
        for nt in x_dict:
            if nt not in out:
                out[nt] = x_dict[nt]
        x_dict = {nt: norm_dict[nt](x) for nt, x in out.items()}
        x_dict = {nt: x.relu() for nt, x in x_dict.items()}
    return x_dict


# ---------------------------------------------------------------------------
# Encoder variants for the A3 ablation. All three produce a per-node-type
# ``x_dict`` with a common channel dim so they slot into the same GNN backbone.
# ---------------------------------------------------------------------------
class IdEncoder(nn.Module):
    """A3 'id' variant — one learnable vector *per node type* (broadcast to
    every node of that type). No column info, no per-node identity, just a
    type-level starting representation. Signal is purely structural.

    We deliberately avoid a per-node embedding table because on rel-hm the
    customer table has ~1.4M rows → a 128-d table would add 180M params and
    dominate the capacity comparison. The type-vector formulation matches the
    "id-only" reading used by the RelBench baselines and by our custom a3."""
    def __init__(self, data: HeteroData, channels: int):
        super().__init__()
        self.type_vec = nn.ParameterDict({
            nt: nn.Parameter(torch.randn(channels) * 0.02)
            for nt in data.node_types
        })

    def forward(self, batch: HeteroData) -> Dict[str, torch.Tensor]:
        return {
            nt: self.type_vec[nt].unsqueeze(0).expand(batch[nt].num_nodes, -1)
            for nt in self.type_vec
        }


class ColumnEncoder(nn.Module):
    """A3 'column' variant — HeteroEncoder over Frame *without* text stypes.
    We only register non-text stypes with the underlying ResNet encoder, so
    the text columns in ``batch.tf_dict`` are silently ignored at forward.
    (HeteroEncoder's per-type ResNet iterates over its configured
    ``col_names_dict``, not over the tf's own dict.)"""
    def __init__(self, data: HeteroData, col_stats_dict, channels: int):
        super().__init__()
        node_to_col_names_dict = {}
        for nt in data.node_types:
            cnd = dict(data[nt].tf.col_names_dict)
            cnd = {st: cols for st, cols in cnd.items()
                   if "text" not in str(st).lower()}
            node_to_col_names_dict[nt] = cnd
        node_to_col_stats = {nt: col_stats_dict[nt] for nt in data.node_types}
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict=node_to_col_names_dict,
            node_to_col_stats=node_to_col_stats,
        )

    def forward(self, batch: HeteroData) -> Dict[str, torch.Tensor]:
        return self.encoder(batch.tf_dict)


class LLMEncoder(nn.Module):
    """A3 'llm' variant — the paper's default HeteroEncoder over all Frame
    stypes including ``text_embedded``. Text columns are already encoded to
    384-d sentence-transformer vectors at data-build time."""
    def __init__(self, data: HeteroData, col_stats_dict, channels: int):
        super().__init__()
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                nt: data[nt].tf.col_names_dict for nt in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )

    def forward(self, batch: HeteroData) -> Dict[str, torch.Tensor]:
        return self.encoder(batch.tf_dict)


# ---------------------------------------------------------------------------
# GNN backbone options for A2.
# ---------------------------------------------------------------------------
def build_gnn(data: HeteroData, kind: str, channels: int, num_layers: int,
              aggr: str = AGGR, num_heads: int = 4):
    """Instantiate the message-passing stack.

    * ``kind='sage'`` -> RelBench's ``HeteroGraphSAGE``
    * ``kind='hgt'``  -> a stack of ``HGTConv`` with a shared metadata
      description (heads×head_channels = channels)
    """
    if kind == "sage":
        return HeteroGraphSAGE(
            node_types=list(data.node_types),
            edge_types=list(data.edge_types),
            channels=channels, aggr=aggr, num_layers=num_layers,
        )
    if kind == "hgt":
        from torch_geometric.nn import HGTConv
        stack = nn.ModuleList()
        metadata = (list(data.node_types), list(data.edge_types))
        for _ in range(num_layers):
            stack.append(HGTConv(channels, channels, metadata,
                                  heads=num_heads))
        return stack
    raise ValueError(f"unknown gnn kind: {kind}")


def run_gnn(gnn, kind: str, x_dict, edge_index_dict):
    """Forward pass for either backbone (with safe handling of empty edges).

    HGT-specific: unlike SAGE we do *not* apply an external ReLU between
    :class:`HGTConv` layers — HGTConv already carries its own multi-head
    attention + residual scheme, and the standard PyG hetero-HGT example
    chains raw conv calls. Adding an external ReLU zeros out too much of the
    type-specific attention output; empirically the model then collapses to a
    constant 0.5 prediction (BCE loss stuck at 0.25 with our pos_weight). We
    still carry-forward node types that no HGT edge touches this batch so the
    stack stays well-defined."""
    if kind == "sage":
        return safe_hetero_forward(gnn, x_dict, edge_index_dict)
    if kind == "hgt":
        ei_filtered = {et: ei for et, ei in edge_index_dict.items()
                       if ei.numel() > 0}
        for conv in gnn:
            new_x = conv(x_dict, ei_filtered)
            for nt in x_dict:
                if nt not in new_x:
                    new_x[nt] = x_dict[nt]
            x_dict = new_x
        return x_dict
    raise ValueError(f"unknown gnn kind: {kind}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    return {
        "roc_auc":   float(roc_auc_score(y_true, y_prob)),
        "auprc":     float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
    }


def find_best_f1(y_true: np.ndarray, y_prob: np.ndarray,
                 thresholds=None) -> Dict[str, float]:
    """Sweep thresholds and return the best F1 with its (precision, recall)
    and threshold. Cheap: ≤ 91 threshold evaluations per call."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)
    best = {"f1": 0.0, "threshold": 0.5, "precision": 0.0, "recall": 0.0}
    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.int64)
        f = f1_score(y_true, y_pred, zero_division=0)
        if f > best["f1"]:
            best = {
                "f1": float(f), "threshold": float(t),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
            }
    return best


def compute_pos_weight_from_task(task) -> float:
    """``n_neg / n_pos`` from the task's train table — for
    :class:`BCEWithLogitsLoss(pos_weight=...)``."""
    df = task.get_table("train").df
    y = df[task.target_col].astype(int).values
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return float(n_neg) / max(n_pos, 1)


# ---------------------------------------------------------------------------
# Training loop — model-agnostic, expects ``model(batch)`` -> logits.
# ---------------------------------------------------------------------------
def train_and_eval(model, target_node: str,
                   train_loader, val_loader,
                   device: str, hb=print, *,
                   epochs: int = EPOCHS,
                   lr: float = LR,
                   pos_weight: Optional[float] = None,
                   max_steps: int = MAX_STEPS_PER_EPOCH):
    from torch.nn import BCEWithLogitsLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if pos_weight is None:
        criterion = BCEWithLogitsLoss()
    else:
        criterion = BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, device=device))

    best_val_auc = -1.0
    best = None
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for i, batch in enumerate(train_loader):
            if i >= max_steps:
                break
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch).squeeze(-1)
            labels = batch[target_node].y.float()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            n_batches += 1

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch).squeeze(-1)
                ys.append(batch[target_node].y.detach().cpu().numpy())
                ps.append(torch.sigmoid(logits).detach().cpu().numpy())
        y_true = np.concatenate(ys)
        y_prob = np.concatenate(ps)
        val = compute_metrics(y_true, y_prob)
        val_best_f1 = find_best_f1(y_true, y_prob)
        hb(f"  ep {ep:>2d}/{epochs} loss={total_loss/max(n_batches,1):.4f} "
           f"AUC={val['roc_auc']:.4f}  best-F1={val_best_f1['f1']:.3f}"
           f"@t={val_best_f1['threshold']:.2f}")

        if val["roc_auc"] > best_val_auc:
            best_val_auc = val["roc_auc"]
            best = {"epoch": ep, "val": val, "val_best_f1": val_best_f1}
    return best


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
