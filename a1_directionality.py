"""Aspect 1 (Message Directionality) using the RelBench-reference
pipeline. Uses PyTorch Frame + sentence-transformer text embeddings, matches
the paper's hyperparameters, and hits ~0.69 val AUC on rel-hm/user-churn (the
paper's own reported value).

Directionality variants:
* **U** — default undirected graph (make_pkey_fkey_graph already adds the
  reverse edges)
* **D** — FK-only (drop every ``rev_`` edge)
* **DirGNN** — two parallel HeteroGraphSAGE stacks (fwd/bwd) with a linear
  merge on the target node.

The D/DirGNN variants use :func:`pipeline.safe_hetero_forward` which carries
representations forward for any node type left isolated by the edge-type
filter — without this, HeteroConv's aggregation crashes on ``None`` messages.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")

from common import setup_paths, log_result, hb
setup_paths()

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData

from pipeline import (
    CHANNELS, NUM_LAYERS, LR, BATCH_SIZE, NUM_NEIGHBORS, EPOCHS, AGGR,
    TEMPORAL_STRATEGY, MAX_STEPS_PER_EPOCH, SEED,
    build_graph_data, build_loaders, safe_hetero_forward,
    train_and_eval, count_params, compute_pos_weight_from_task,
)
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder


DATASETS = [("rel-hm", "user-churn")]
DIRECTIONS = ["U", "D", "DirGNN"]


class GNNModel(nn.Module):
    """HeteroEncoder + HeteroTemporalEncoder + HeteroGraphSAGE + linear head,
    with a directionality switch."""
    def __init__(self, data: HeteroData, col_stats_dict, num_classes: int,
                 target_node: str, directionality: str,
                 channels: int = CHANNELS, num_layers: int = NUM_LAYERS,
                 aggr: str = AGGR):
        super().__init__()
        assert directionality in ("U", "D", "DirGNN")
        self.directionality = directionality
        self.target_node = target_node

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

        edge_types = list(data.edge_types)
        # Always build with the *full* edge-type set — we filter at forward
        # time. That keeps SAGEConv weights consistent between U/D/DirGNN
        # comparisons (i.e., only the routing differs, not the module count).
        self.gnn = HeteroGraphSAGE(
            node_types=list(data.node_types), edge_types=edge_types,
            channels=channels, aggr=aggr, num_layers=num_layers,
        )
        if directionality == "DirGNN":
            self.gnn_bwd = HeteroGraphSAGE(
                node_types=list(data.node_types), edge_types=edge_types,
                channels=channels, aggr=aggr, num_layers=num_layers,
            )
            self.merge = nn.Linear(2 * channels, channels)
        else:
            self.gnn_bwd = None
            self.merge = None
        self.head = nn.Linear(channels, num_classes)

    def _apply_encoder(self, batch: HeteroData):
        x_dict = self.encoder(batch.tf_dict)
        rel_time_dict = self.temporal_encoder(
            batch[self.target_node].seed_time,
            batch.time_dict, batch.batch_dict,
        )
        for nt, rel in rel_time_dict.items():
            x_dict[nt] = x_dict[nt] + rel
        return x_dict

    def forward(self, batch: HeteroData) -> torch.Tensor:
        x_dict = self._apply_encoder(batch)
        if self.directionality == "U":
            out = safe_hetero_forward(self.gnn, x_dict, batch.edge_index_dict)
        elif self.directionality == "D":
            ei_fwd = {et: ei for et, ei in batch.edge_index_dict.items()
                      if not et[1].startswith("rev_")}
            out = safe_hetero_forward(self.gnn, x_dict, ei_fwd)
        else:  # DirGNN
            ei_fwd = {et: ei for et, ei in batch.edge_index_dict.items()
                      if not et[1].startswith("rev_")}
            ei_bwd = {et: ei for et, ei in batch.edge_index_dict.items()
                      if et[1].startswith("rev_")}
            out_fwd = safe_hetero_forward(self.gnn, x_dict, ei_fwd)
            out_bwd = safe_hetero_forward(self.gnn_bwd, x_dict, ei_bwd)
            out = {}
            for nt in x_dict:
                if nt in out_fwd and nt in out_bwd:
                    out[nt] = F.relu(self.merge(
                        torch.cat([out_fwd[nt], out_bwd[nt]], dim=-1)))
                elif nt in out_fwd:
                    out[nt] = out_fwd[nt]
                else:
                    out[nt] = out_bwd[nt]
        return self.head(out[self.target_node][:batch[self.target_node].batch_size])


def main():
    hb("=== A1 start ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ds_name, tk_name in DATASETS:
        t0 = time.time()
        data, col_stats_dict, task, _ = build_graph_data(ds_name, tk_name, device, hb=hb)
        hb(f"  data built in {time.time()-t0:.1f}s")
        pos_w = compute_pos_weight_from_task(task)
        hb(f"  pos_weight = {pos_w:.3f}")
        for direction in DIRECTIONS:
            hb(f"  === direction={direction} ===")
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)
            t_run = time.time()
            train_loader, val_loader = build_loaders(data, task)
            model = GNNModel(
                data, col_stats_dict, num_classes=1,
                target_node=task.entity_table, directionality=direction,
            ).to(device)
            n_params = count_params(model)
            hb(f"  {direction} params={n_params:,}")
            best = train_and_eval(
                model, task.entity_table, train_loader, val_loader,
                device=device, hb=hb, pos_weight=pos_w,
            )
            log_result("a1_directionality", {
                "dataset": ds_name, "task": tk_name,
                "directionality": direction,
                "n_params": n_params,
                "config": {
                    "channels": CHANNELS, "num_layers": NUM_LAYERS,
                    "lr": LR, "batch_size": BATCH_SIZE,
                    "num_neighbors": NUM_NEIGHBORS, "epochs": EPOCHS,
                    "aggr": AGGR, "temporal_strategy": TEMPORAL_STRATEGY,
                    "seed": SEED, "max_steps_per_epoch": MAX_STEPS_PER_EPOCH,
                    "pos_weight": pos_w,
                },
                "best_epoch": best["epoch"],
                "val": best["val"],
                "val_best_f1": best["val_best_f1"],
                "wall_seconds": time.time() - t_run,
            })
            hb(f"  direction={direction} best AUC={best['val']['roc_auc']:.4f} "
               f"best-F1={best['val_best_f1']['f1']:.3f}")
    hb("=== A1 done ===")


if __name__ == "__main__":
    main()
