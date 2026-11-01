"""Aspect 4 with **mean** aggregation on ``SAGEConv`` — spec-compliant
version of :mod:`a4_depth`. Guidelines specify "GraphSAGE (with mean
aggregation)" for A4; our original A4 used ``aggr='sum'`` (RelBench
paper default). This rerun switches to ``aggr='mean'`` and additionally
records **per-layer MAD** (mean pairwise cosine distance across target
seeds) at every depth so we have the similarity-measure evidence the
guidelines require.

Runs the standard depth set and skip variants that the reference-sum
runs already covered, but with mean-agg:

  * baseline depths: [2, 3, 4, 5, 6, 7, 8]  (skipping L=16 — memory)
  * skip depths:     [4, 8]                 (skipping L=16 — memory)
  * on both datasets

Results append to ``results/a4_mean/results.jsonl``.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")

from common import setup_paths, log_result, hb
setup_paths()

import numpy as np
import torch
import torch.nn as nn

from torch_geometric.data import HeteroData

from pipeline import (
    CHANNELS, LR, EPOCHS, TEMPORAL_STRATEGY, MAX_STEPS_PER_EPOCH, SEED,
    build_graph_data, train_and_eval, count_params, compute_pos_weight_from_task,
)
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder


AGGR = "mean"        # <-- spec
BATCH_SIZE = 128
BASELINE_DEPTHS = [2, 3, 4, 5, 6, 7, 8]
SKIP_DEPTHS     = [4, 8]
TASKS = [
    ("rel-hm",    "user-churn"),
    ("rel-stack", "user-engagement"),
]


def _capped_neighbors(n_layers: int):
    base = [128, 32]
    return base[:min(2, n_layers)] + [5] * max(0, n_layers - 2)


class DepthMeanModel(nn.Module):
    """Depth-configurable + skip-configurable + records per-layer MAD when
    ``self._collect_mad`` is set on the module."""
    def __init__(self, data: HeteroData, col_stats_dict, num_classes: int,
                 target_node: str, n_layers: int, skip: bool,
                 channels: int = CHANNELS):
        super().__init__()
        self.target_node = target_node
        self.skip = skip
        self.n_layers = n_layers
        self._collect_mad = False       # toggle from outside during eval
        self._last_mad = None

        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                nt: data[nt].tf.col_names_dict for nt in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        node_types_with_time = [nt for nt in data.node_types if "time" in data[nt]]
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=node_types_with_time, channels=channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=list(data.node_types),
            edge_types=list(data.edge_types),
            channels=channels, aggr=AGGR, num_layers=n_layers,
        )
        self.head = nn.Linear(channels, num_classes)

    def forward(self, batch: HeteroData) -> torch.Tensor:
        x_dict = self.encoder(batch.tf_dict)
        rel_time_dict = self.temporal_encoder(
            batch[self.target_node].seed_time,
            batch.time_dict, batch.batch_dict,
        )
        for nt, rel in rel_time_dict.items():
            x_dict[nt] = x_dict[nt] + rel
        ei_filtered = {et: ei for et, ei in batch.edge_index_dict.items()
                       if ei.numel() > 0}

        # per-layer target reps (only if collecting MAD)
        bsz = batch[self.target_node].batch_size
        per_layer_reps = []
        if self._collect_mad:
            per_layer_reps.append(x_dict[self.target_node][:bsz].detach())

        for conv, norm_dict in zip(self.gnn.convs, self.gnn.norms):
            residual = x_dict
            out = conv(x_dict, ei_filtered)
            for nt in x_dict:
                if nt not in out:
                    out[nt] = x_dict[nt]
            if self.skip:
                out = {nt: v + residual[nt] for nt, v in out.items()}
            x_dict = {nt: norm_dict[nt](v) for nt, v in out.items()}
            x_dict = {nt: v.relu() for nt, v in x_dict.items()}
            if self._collect_mad:
                per_layer_reps.append(x_dict[self.target_node][:bsz].detach())

        if self._collect_mad and per_layer_reps:
            self._last_mad = [_mad(r) for r in per_layer_reps]

        return self.head(x_dict[self.target_node][:bsz])


def _mad(x: torch.Tensor) -> float:
    """Mean pairwise cosine distance across a sample of node embeddings.
    Uses at most 512 nodes to keep it O(512^2)."""
    if x.numel() == 0: return 0.0
    if x.shape[0] > 512:
        idx = torch.randperm(x.shape[0], device=x.device)[:512]
        x = x[idx]
    n = torch.nn.functional.normalize(x, dim=1)
    sim = n @ n.t()
    # off-diagonal mean
    mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    dist = 1.0 - sim[mask]
    return float(dist.mean().item())


@torch.no_grad()
def collect_mad_on_val(model, val_loader, target_node, device):
    """Run one val batch with MAD collection on."""
    model.eval()
    model._collect_mad = True
    for batch in val_loader:
        batch = batch.to(device)
        _ = model(batch)
        break
    model._collect_mad = False
    return model._last_mad or []


def run_one(data, col_stats_dict, task, ds_name, tk_name,
             pos_w, n_layers, skip, device):
    tag = f"{ds_name} L={n_layers} skip={skip}  aggr=mean"
    hb(f"  === {tag} ===")
    torch.cuda.empty_cache()
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    t_run = time.time()

    from relbench.modeling.graph import get_node_train_table_input
    from torch_geometric.loader import NeighborLoader
    neighbors = _capped_neighbors(n_layers)
    hb(f"  fan-out per layer = {neighbors}, batch={BATCH_SIZE}")
    train_ti = get_node_train_table_input(
        table=task.get_table("train"), task=task)
    val_ti = get_node_train_table_input(
        table=task.get_table("val"), task=task)
    train_loader = NeighborLoader(
        data, num_neighbors=neighbors, time_attr="time",
        input_nodes=train_ti.nodes, input_time=train_ti.time,
        transform=train_ti.transform, batch_size=BATCH_SIZE,
        temporal_strategy=TEMPORAL_STRATEGY, shuffle=True, num_workers=0,
    )
    val_loader = NeighborLoader(
        data, num_neighbors=neighbors, time_attr="time",
        input_nodes=val_ti.nodes, input_time=val_ti.time,
        transform=val_ti.transform, batch_size=BATCH_SIZE,
        temporal_strategy=TEMPORAL_STRATEGY, shuffle=False, num_workers=0,
    )
    model = DepthMeanModel(
        data, col_stats_dict, num_classes=1,
        target_node=task.entity_table,
        n_layers=n_layers, skip=skip,
    ).to(device)
    n_params = count_params(model)
    hb(f"  {tag} params={n_params:,}")
    try:
        best = train_and_eval(
            model, task.entity_table, train_loader, val_loader,
            device=device, hb=hb, pos_weight=pos_w,
        )
    except torch.cuda.OutOfMemoryError as e:
        hb(f"  {tag} OOM: {e}")
        del model, train_loader, val_loader
        torch.cuda.empty_cache()
        return

    layer_mad = collect_mad_on_val(model, val_loader, task.entity_table, device)
    hb(f"  {tag} layer_mad = {[round(v, 5) for v in layer_mad]}")

    log_result("a4_mean", {
        "dataset": ds_name, "task": tk_name,
        "n_layers": n_layers, "skip_connections": skip,
        "aggr": AGGR,
        "n_params": n_params,
        "layer_mad": layer_mad,
        "config": {
            "channels": CHANNELS, "num_layers": n_layers,
            "lr": LR, "batch_size": BATCH_SIZE,
            "num_neighbors": neighbors, "epochs": EPOCHS,
            "aggr": AGGR, "temporal_strategy": TEMPORAL_STRATEGY,
            "seed": SEED, "pos_weight": pos_w,
        },
        "best_epoch": best["epoch"],
        "val": best["val"],
        "val_best_f1": best["val_best_f1"],
        "wall_seconds": time.time() - t_run,
    })
    hb(f"  {tag} best AUC={best['val']['roc_auc']:.4f} "
       f"best-F1={best['val_best_f1']['f1']:.3f}")
    del model, train_loader, val_loader
    torch.cuda.empty_cache()


def main():
    hb("=== A4-MEAN + MAD start ===")
    hb(f"  PYTORCH_CUDA_ALLOC_CONF = {os.environ.get('PYTORCH_CUDA_ALLOC_CONF','<unset>')}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ds_name, tk_name in TASKS:
        t0 = time.time()
        data, col_stats_dict, task, _ = build_graph_data(ds_name, tk_name, device, hb=hb)
        hb(f"  {ds_name} data built in {time.time()-t0:.1f}s")
        pos_w = compute_pos_weight_from_task(task)
        hb(f"  pos_weight = {pos_w:.3f}")
        for d in BASELINE_DEPTHS:
            run_one(data, col_stats_dict, task, ds_name, tk_name,
                    pos_w, d, False, device)
        for d in SKIP_DEPTHS:
            run_one(data, col_stats_dict, task, ds_name, tk_name,
                    pos_w, d, True, device)
        del data, col_stats_dict, task
        torch.cuda.empty_cache()
    hb("=== A4-MEAN + MAD done ===")


if __name__ == "__main__":
    main()
