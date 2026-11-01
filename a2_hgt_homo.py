"""A2 HGT-Homo variant on the RelBench pipeline — spec-required
(guideline: "HGT (disable the heterogeneity by building HeteroData such
that all nodes have the same type, and all edges have same type)").

Structure:
* Keep per-node-type Frame encoders (each table has a different schema — we
  can't share one encoder), so the initial-representation step is still
  heterogeneous.
* After encoding + temporal, concatenate every node type's embedding into a
  single tensor and homogenise the edge dictionary (all edges become one
  type, all nodes become one type). Record the target-node's global index
  range so we can slice out its logits at the end.
* Run a stack of ``HGTConv`` on this synthetic single-type HeteroData —
  metadata is ``(["node"], [("node", "edge", "node")])`` — matches the
  "disable heterogeneity" phrasing in the guidelines.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")

from common import setup_paths, log_result, hb
setup_paths()

import torch
import torch.nn as nn

from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv

from pipeline import (
    CHANNELS, NUM_LAYERS, LR, BATCH_SIZE, NUM_NEIGHBORS, EPOCHS,
    TEMPORAL_STRATEGY, SEED,
    build_graph_data, build_loaders,
    train_and_eval, count_params, compute_pos_weight_from_task,
)
from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder


DATASETS = [
    ("rel-hm", "user-churn"),
    ("rel-stack", "user-engagement"),
]
NUM_HEADS = 4


class HGTHomo(nn.Module):
    def __init__(self, data: HeteroData, col_stats_dict, num_classes: int,
                 target_node: str, channels: int = CHANNELS,
                 num_layers: int = NUM_LAYERS, num_heads: int = NUM_HEADS):
        super().__init__()
        self.target_node = target_node
        self.node_types = list(data.node_types)

        # heterogeneous encoders (unavoidable — per-type Frame schemas)
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

        # homogeneous HGT — single node type, single edge type
        metadata = (["node"], [("node", "edge", "node")])
        self.convs = nn.ModuleList([
            HGTConv(channels, channels, metadata, heads=num_heads)
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
        x_homo = torch.cat(parts, dim=0)   # (total_nodes, channels)

        # 3) collapse edges: shift indices, concat, dedupe not needed
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

        # 4) run HGT on single-type dicts
        x_dict_h = {"node": x_homo}
        ei_dict_h = {("node", "edge", "node"): ei_homo}
        for conv in self.convs:
            new_x = conv(x_dict_h, ei_dict_h)
            if "node" not in new_x:
                new_x["node"] = x_dict_h["node"]
            x_dict_h = new_x

        # 5) slice target-node seeds
        bsz = batch[self.target_node].batch_size
        tgt_start = offsets[self.target_node]
        target_out = x_dict_h["node"][tgt_start : tgt_start + bsz]
        return self.head(target_out)


def main():
    hb("=== A2-HGT-Homo start ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ds_name, tk_name in DATASETS:
        t0 = time.time()
        data, col_stats_dict, task, _ = build_graph_data(ds_name, tk_name, device, hb=hb)
        hb(f"  data built in {time.time()-t0:.1f}s")
        pos_w = compute_pos_weight_from_task(task)
        hb(f"  pos_weight = {pos_w:.3f}")

        hb(f"  === HGT-Homo on {ds_name} ===")
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        t_run = time.time()
        train_loader, val_loader = build_loaders(data, task)
        model = HGTHomo(
            data, col_stats_dict, num_classes=1,
            target_node=task.entity_table,
        ).to(device)
        n_params = count_params(model)
        hb(f"  HGT-Homo params={n_params:,}")
        best = train_and_eval(
            model, task.entity_table, train_loader, val_loader,
            device=device, hb=hb, pos_weight=pos_w,
        )
        # append to a2_heterogeneity/ so the A2 comparison table has all variants
        log_result("a2_heterogeneity", {
            "dataset": ds_name, "task": tk_name,
            "model": "HGT-Homo",
            "n_params": n_params,
            "config": {
                "channels": CHANNELS, "num_layers": NUM_LAYERS,
                "num_heads": NUM_HEADS,
                "lr": LR, "batch_size": BATCH_SIZE,
                "num_neighbors": NUM_NEIGHBORS, "epochs": EPOCHS,
                "temporal_strategy": TEMPORAL_STRATEGY,
                "seed": SEED, "pos_weight": pos_w,
                "note": "single-type collapse: all node types -> 'node', all edges -> 'edge'",
            },
            "best_epoch": best["epoch"],
            "val": best["val"],
            "val_best_f1": best["val_best_f1"],
            "wall_seconds": time.time() - t_run,
        })
        hb(f"  HGT-Homo best AUC={best['val']['roc_auc']:.4f} "
           f"best-F1={best['val_best_f1']['f1']:.3f}")
        del model, train_loader, val_loader
        torch.cuda.empty_cache()
    hb("=== A2-HGT-Homo done ===")


if __name__ == "__main__":
    main()
