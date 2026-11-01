"""Aspect 4 (Oversmoothing + mitigation) using the RelBench-reference pipeline.

Depth sweep on rel-hm/user-churn with the RelBench pipeline. Two variants per
depth: (a) no-skip (vanilla), (b) skip-connections (residual after each
HeteroConv layer). We use the same shared encoder / temporal encoder / linear
head as A1-U so only the message-passing depth changes.

Skip implementation: after each :class:`HeteroConv` layer, add the previous
representation back before LayerNorm+ReLU. Equivalent to identity residuals.
(This matches the "skip" variant in our custom A4 pipeline.)

Depths: {2, 3, 4, 5, 6, 7, 8, 16}. Skip variant only at {4, 8, 16} — the
lower depths don't oversmooth enough for skip to matter, we keep them
baseline-only to save GPU.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")

from common import setup_paths, log_result, hb
setup_paths()

import torch
import torch.nn as nn

from torch_geometric.data import HeteroData

from pipeline import (
    CHANNELS, LR, BATCH_SIZE, NUM_NEIGHBORS, EPOCHS, AGGR,
    TEMPORAL_STRATEGY, MAX_STEPS_PER_EPOCH, SEED,
    build_graph_data, build_loaders,
    train_and_eval, count_params, compute_pos_weight_from_task,
)
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder


DATASETS = [("rel-hm", "user-churn")]
BASELINE_DEPTHS = [2, 3, 4, 5, 6, 7, 8, 16]
SKIP_DEPTHS = [4, 8, 16]


class DepthModel(nn.Module):
    """RelBench pipeline with configurable depth and optional residual skip
    connections. Neighbor fan-out is ``NUM_NEIGHBORS / 2**i`` at layer ``i``,
    same as A1 — with depth 16 the outermost layers use very few neighbors
    (which is fine and matches the paper's default sampling strategy)."""
    def __init__(self, data: HeteroData, col_stats_dict, num_classes: int,
                 target_node: str, n_layers: int, skip: bool,
                 channels: int = CHANNELS, aggr: str = AGGR):
        super().__init__()
        self.target_node = target_node
        self.skip = skip
        self.n_layers = n_layers

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
            channels=channels, aggr=aggr, num_layers=n_layers,
        )
        self.head = nn.Linear(channels, num_classes)

    def _apply_encoder(self, batch):
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
        ei_filtered = {et: ei for et, ei in batch.edge_index_dict.items()
                       if ei.numel() > 0}
        # Manual HeteroConv loop so we can splice in a residual skip.
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
        return self.head(x_dict[self.target_node][:batch[self.target_node].batch_size])


def _neighbors_for_depth(n_layers: int):
    return [max(1, int(NUM_NEIGHBORS / 2**i)) for i in range(n_layers)]


def main():
    hb("=== A4 start ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ds_name, tk_name in DATASETS:
        t0 = time.time()
        data, col_stats_dict, task, _ = build_graph_data(ds_name, tk_name, device, hb=hb)
        hb(f"  data built in {time.time()-t0:.1f}s")
        pos_w = compute_pos_weight_from_task(task)
        hb(f"  pos_weight = {pos_w:.3f}")

        combos = ([(d, False) for d in BASELINE_DEPTHS] +
                  [(d, True)  for d in SKIP_DEPTHS])
        for n_layers, skip in combos:
            tag = f"L={n_layers} skip={skip}"
            hb(f"  === {tag} ===")
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)
            t_run = time.time()
            train_loader, val_loader = build_loaders(
                data, task, num_layers=n_layers)
            model = DepthModel(
                data, col_stats_dict, num_classes=1,
                target_node=task.entity_table,
                n_layers=n_layers, skip=skip,
            ).to(device)
            n_params = count_params(model)
            hb(f"  {tag} params={n_params:,}")
            best = train_and_eval(
                model, task.entity_table, train_loader, val_loader,
                device=device, hb=hb, pos_weight=pos_w,
            )
            log_result("a4_depth", {
                "dataset": ds_name, "task": tk_name,
                "n_layers": n_layers, "skip_connections": skip,
                "n_params": n_params,
                "config": {
                    "channels": CHANNELS, "num_layers": n_layers,
                    "lr": LR, "batch_size": BATCH_SIZE,
                    "num_neighbors": _neighbors_for_depth(n_layers),
                    "epochs": EPOCHS,
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
    hb("=== A4 done ===")


if __name__ == "__main__":
    main()
