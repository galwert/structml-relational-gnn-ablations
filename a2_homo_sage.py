"""A2 HomoSAGE variant on the RelBench pipeline — same pattern as
:mod:`a2_hgt_homo` but with :class:`SAGEConv` instead of
:class:`HGTConv`, so we can complete the 2×2 Hetero/Homo × SAGE/HGT
matrix under the RelBench pipeline.

Structure:
* Keep the per-node-type Frame encoders (each table has a different
  schema; sharing one encoder isn't possible).
* Collapse the graph after encoding: concat every node type's
  embedding into a single tensor, shift + concat edge indices into a
  single (2, E) tensor.
* Run a stack of homogeneous ``SAGEConv`` on the collapsed graph
  (this is the "disable heterogeneity" story --- all nodes and edges
  share one message function).
* Slice out the target-node embeddings and pass through a linear head.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")

from common import setup_paths, log_result, hb
setup_paths()

import torch
import torch.nn as nn

from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv

from pipeline import (
    CHANNELS, NUM_LAYERS, LR, BATCH_SIZE, NUM_NEIGHBORS, EPOCHS, AGGR,
    TEMPORAL_STRATEGY, SEED,
    build_graph_data, build_loaders,
    train_and_eval, count_params, compute_pos_weight_from_task,
)
from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder


DATASETS = [
    ("rel-hm", "user-churn"),
    ("rel-stack", "user-engagement"),
]


class HomoSAGEModel(nn.Module):
    def __init__(self, data: HeteroData, col_stats_dict, num_classes: int,
                 target_node: str, channels: int = CHANNELS,
                 num_layers: int = NUM_LAYERS, aggr: str = AGGR):
        super().__init__()
        self.target_node = target_node

        # heterogeneous encoders (per-schema Frame — unavoidable)
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

        # homogeneous SAGE stack — one message function for all node/edge types
        self.convs = nn.ModuleList([
            SAGEConv((channels, channels), channels, aggr=aggr)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(channels, num_classes)

    def forward(self, batch: HeteroData) -> torch.Tensor:
        # 1) heterogeneous encode
        x_dict = self.encoder(batch.tf_dict)
        rel_time_dict = self.temporal_encoder(
            batch[self.target_node].seed_time,
            batch.time_dict, batch.batch_dict,
        )
        for nt, rel in rel_time_dict.items():
            x_dict[nt] = x_dict[nt] + rel

        # 2) collapse to single-type: concat + record offsets
        offsets = {}
        parts = []
        cum = 0
        for nt in batch.node_types:
            offsets[nt] = cum
            parts.append(x_dict[nt])
            cum += batch[nt].num_nodes
        x_homo = torch.cat(parts, dim=0)

        # 3) collapse edges: shift indices and concat
        src_all, dst_all = [], []
        for et in batch.edge_types:
            ei = batch[et].edge_index
            if ei.numel() == 0: continue
            src_type, _, dst_type = et
            src_all.append(ei[0] + offsets[src_type])
            dst_all.append(ei[1] + offsets[dst_type])
        if src_all:
            ei_homo = torch.stack([torch.cat(src_all), torch.cat(dst_all)])
        else:
            ei_homo = torch.zeros((2, 0), dtype=torch.long, device=x_homo.device)

        # 4) run homogeneous SAGE on the collapsed graph
        x = x_homo
        for conv in self.convs:
            x = conv(x, ei_homo).relu()

        # 5) slice target-node seeds
        bsz = batch[self.target_node].batch_size
        tgt_start = offsets[self.target_node]
        target_out = x[tgt_start : tgt_start + bsz]
        return self.head(target_out)


def main():
    hb("=== A2-HomoSAGE start ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ds_name, tk_name in DATASETS:
        t0 = time.time()
        data, col_stats_dict, task, _ = build_graph_data(ds_name, tk_name, device, hb=hb)
        hb(f"  data built in {time.time()-t0:.1f}s")
        pos_w = compute_pos_weight_from_task(task)
        hb(f"  pos_weight = {pos_w:.3f}")

        hb(f"  === HomoSAGE on {ds_name} ===")
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        t_run = time.time()
        train_loader, val_loader = build_loaders(data, task)
        model = HomoSAGEModel(
            data, col_stats_dict, num_classes=1,
            target_node=task.entity_table,
        ).to(device)
        n_params = count_params(model)
        hb(f"  HomoSAGE params={n_params:,}")
        best = train_and_eval(
            model, task.entity_table, train_loader, val_loader,
            device=device, hb=hb, pos_weight=pos_w,
        )
        log_result("a2_heterogeneity", {
            "dataset": ds_name, "task": tk_name,
            "model": "HomoSAGE",
            "n_params": n_params,
            "config": {
                "channels": CHANNELS, "num_layers": NUM_LAYERS,
                "lr": LR, "batch_size": BATCH_SIZE,
                "num_neighbors": NUM_NEIGHBORS, "epochs": EPOCHS,
                "aggr": AGGR, "temporal_strategy": TEMPORAL_STRATEGY,
                "seed": SEED, "pos_weight": pos_w,
                "note": "single-type collapse: all node types -> 'node', all edges -> 'edge'; SAGEConv stack",
            },
            "best_epoch": best["epoch"],
            "val": best["val"],
            "val_best_f1": best["val_best_f1"],
            "wall_seconds": time.time() - t_run,
        })
        hb(f"  HomoSAGE best AUC={best['val']['roc_auc']:.4f} "
           f"best-F1={best['val_best_f1']['f1']:.3f}")
        del model, train_loader, val_loader
        torch.cuda.empty_cache()
    hb("=== A2-HomoSAGE done ===")


if __name__ == "__main__":
    main()
