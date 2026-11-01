"""Aspect 2 (Heterogeneity) using the RelBench-reference pipeline.

Two variants on the same Frame-based encoder + temporal encoder:

* **HeteroSAGE** — the paper's default: ``HeteroGraphSAGE`` (per-edge-type
  SAGEConv with sum aggregation across relations). Identical to A1-U.
* **HGT** — swap the message-passing stack for :class:`torch_geometric.nn.HGTConv`
  (type-aware attention). Same channels/num_layers/temporal encoder — only the
  routing changes.

We drop the "Homo" variants (HomoSAGE / HGT-Homo) for the RelBench pipeline
because collapsing to a single node type doesn't play with per-type
PyTorch-Frame encoders (each type has its own column schema). Homogeneous
numbers on rel-hm come from the custom pipeline (A2 baseline in results/a2/).
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")

from common import setup_paths, log_result, hb
setup_paths()

import torch
import torch.nn as nn

from torch_geometric.data import HeteroData

from pipeline import (
    CHANNELS, NUM_LAYERS, LR, BATCH_SIZE, NUM_NEIGHBORS, EPOCHS, AGGR,
    TEMPORAL_STRATEGY, MAX_STEPS_PER_EPOCH, SEED,
    build_graph_data, build_loaders, build_gnn, run_gnn,
    train_and_eval, count_params, compute_pos_weight_from_task,
)
from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder


DATASETS = [("rel-hm", "user-churn")]
MODELS = ["sage", "hgt"]  # names logged as HeteroSAGE / HGT-Hetero


class Backbone(nn.Module):
    """Same encoder-plus-GNN structure as :class:`a1_directionality.GNNModel`, but the
    GNN stack is configurable — SAGE or HGT."""
    def __init__(self, data: HeteroData, col_stats_dict, num_classes: int,
                 target_node: str, gnn_kind: str,
                 channels: int = CHANNELS, num_layers: int = NUM_LAYERS,
                 aggr: str = AGGR):
        super().__init__()
        self.gnn_kind = gnn_kind
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
        self.gnn = build_gnn(data, gnn_kind, channels, num_layers, aggr=aggr)
        self.head = nn.Linear(channels, num_classes)

    def forward(self, batch: HeteroData) -> torch.Tensor:
        x_dict = self.encoder(batch.tf_dict)
        rel_time_dict = self.temporal_encoder(
            batch[self.target_node].seed_time,
            batch.time_dict, batch.batch_dict,
        )
        for nt, rel in rel_time_dict.items():
            x_dict[nt] = x_dict[nt] + rel
        out = run_gnn(self.gnn, self.gnn_kind, x_dict, batch.edge_index_dict)
        return self.head(out[self.target_node][:batch[self.target_node].batch_size])


def _pretty(kind: str) -> str:
    return {"sage": "HeteroSAGE", "hgt": "HGT-Hetero"}[kind]


def main():
    hb("=== A2 start ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ds_name, tk_name in DATASETS:
        t0 = time.time()
        data, col_stats_dict, task, _ = build_graph_data(ds_name, tk_name, device, hb=hb)
        hb(f"  data built in {time.time()-t0:.1f}s")
        pos_w = compute_pos_weight_from_task(task)
        hb(f"  pos_weight = {pos_w:.3f}")
        for kind in MODELS:
            name = _pretty(kind)
            hb(f"  === model={name} ===")
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)
            t_run = time.time()
            train_loader, val_loader = build_loaders(data, task)
            model = Backbone(
                data, col_stats_dict, num_classes=1,
                target_node=task.entity_table, gnn_kind=kind,
            ).to(device)
            n_params = count_params(model)
            hb(f"  {name} params={n_params:,}")
            best = train_and_eval(
                model, task.entity_table, train_loader, val_loader,
                device=device, hb=hb, pos_weight=pos_w,
            )
            log_result("a2_heterogeneity", {
                "dataset": ds_name, "task": tk_name,
                "model": name,
                "n_params": n_params,
                "config": {
                    "channels": CHANNELS, "num_layers": NUM_LAYERS,
                    "lr": LR, "batch_size": BATCH_SIZE,
                    "num_neighbors": NUM_NEIGHBORS, "epochs": EPOCHS,
                    "aggr": AGGR, "temporal_strategy": TEMPORAL_STRATEGY,
                    "seed": SEED, "pos_weight": pos_w,
                },
                "best_epoch": best["epoch"],
                "val": best["val"],
                "val_best_f1": best["val_best_f1"],
                "wall_seconds": time.time() - t_run,
            })
            hb(f"  {name} best AUC={best['val']['roc_auc']:.4f} "
               f"best-F1={best['val_best_f1']['f1']:.3f}")
    hb("=== A2 done ===")


if __name__ == "__main__":
    main()
