"""Aspect 3 (Node Feature Encoding) using the RelBench-reference pipeline.

Three encoders, same GNN backbone (HeteroGraphSAGE):

* **id** — per-node learnable embedding table. No column info at all; the
  signal is entirely structural. This is our ``a3`` custom pipeline's ``id``
  variant reproduced on top of the RelBench graph.
* **column** — HeteroEncoder over PyTorch-Frame *without* text stypes. The
  encoder sees numerical + categorical + timestamp columns but the H&M
  free-text (``detail_desc``, product names, ...) is dropped.
* **llm** — HeteroEncoder over all Frame stypes including
  ``text_embedded``. Text columns were embedded at data-build time with the
  MiniLM sentence-transformer. This is the paper's default.

The comparison isolates *how much of the rel-hm signal is text-driven*. On
our custom pipeline text is missing entirely — the column variant is a fair
apples-to-apples check of "how much do the text columns actually buy".
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
    build_graph_data, build_loaders, safe_hetero_forward,
    train_and_eval, count_params, compute_pos_weight_from_task,
    IdEncoder, ColumnEncoder, LLMEncoder,
)
from relbench.modeling.nn import HeteroGraphSAGE, HeteroTemporalEncoder


DATASETS = [("rel-hm", "user-churn")]
ENCODERS = ["id", "column", "llm"]


class EncoderAblation(nn.Module):
    def __init__(self, data: HeteroData, col_stats_dict, num_classes: int,
                 target_node: str, encoder_kind: str,
                 channels: int = CHANNELS, num_layers: int = NUM_LAYERS,
                 aggr: str = AGGR):
        super().__init__()
        self.encoder_kind = encoder_kind
        self.target_node = target_node

        if encoder_kind == "id":
            self.encoder = IdEncoder(data, channels)
        elif encoder_kind == "column":
            self.encoder = ColumnEncoder(data, col_stats_dict, channels)
        elif encoder_kind == "llm":
            self.encoder = LLMEncoder(data, col_stats_dict, channels)
        else:
            raise ValueError(f"unknown encoder: {encoder_kind}")

        # Temporal encoder — the id variant has no seed_time signal in
        # x_dict, so we still add rel-time here to give the model a temporal
        # anchor. (Otherwise id-only can't learn timing at all.)
        node_types_with_time = [nt for nt in data.node_types if "time" in data[nt]]
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=node_types_with_time, channels=channels,
        )
        self.gnn = HeteroGraphSAGE(
            node_types=list(data.node_types),
            edge_types=list(data.edge_types),
            channels=channels, aggr=aggr, num_layers=num_layers,
        )
        self.head = nn.Linear(channels, num_classes)

    def forward(self, batch: HeteroData) -> torch.Tensor:
        x_dict = self.encoder(batch)
        rel_time_dict = self.temporal_encoder(
            batch[self.target_node].seed_time,
            batch.time_dict, batch.batch_dict,
        )
        for nt, rel in rel_time_dict.items():
            if nt in x_dict:
                x_dict[nt] = x_dict[nt] + rel
        out = safe_hetero_forward(self.gnn, x_dict, batch.edge_index_dict)
        return self.head(out[self.target_node][:batch[self.target_node].batch_size])


def main():
    hb("=== A3 start ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ds_name, tk_name in DATASETS:
        t0 = time.time()
        data, col_stats_dict, task, _ = build_graph_data(ds_name, tk_name, device, hb=hb)
        hb(f"  data built in {time.time()-t0:.1f}s")
        pos_w = compute_pos_weight_from_task(task)
        hb(f"  pos_weight = {pos_w:.3f}")
        for enc in ENCODERS:
            hb(f"  === encoder={enc} ===")
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)
            t_run = time.time()
            train_loader, val_loader = build_loaders(data, task)
            model = EncoderAblation(
                data, col_stats_dict, num_classes=1,
                target_node=task.entity_table, encoder_kind=enc,
            ).to(device)
            n_params = count_params(model)
            hb(f"  encoder={enc} params={n_params:,}")
            best = train_and_eval(
                model, task.entity_table, train_loader, val_loader,
                device=device, hb=hb, pos_weight=pos_w,
            )
            log_result("a3_encoders", {
                "dataset": ds_name, "task": tk_name,
                "encoder": enc,
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
            hb(f"  encoder={enc} best AUC={best['val']['roc_auc']:.4f} "
               f"best-F1={best['val_best_f1']['f1']:.3f}")
    hb("=== A3 done ===")


if __name__ == "__main__":
    main()
