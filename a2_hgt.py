"""Standalone A2 re-run for the HGT-Hetero variant only.

The first A2 run had HGT-Hetero stuck at AUC=0.5000 flat: the training
loss plateau (0.251) corresponded to a constant-output model, caused by an
external ``.relu()`` between HGTConv layers zeroing out the type-specific
attention output. That ReLU is removed in :func:`pipeline.run_gnn`; this
script reruns HGT-Hetero with the fix and appends to
``results/a2_heterogeneity/results.jsonl`` alongside the good HeteroSAGE row.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")

from common import setup_paths, log_result, hb
setup_paths()

import torch

from a2_heterogeneity import Backbone, _pretty  # reuse the model class from a2_heterogeneity
from pipeline import (
    CHANNELS, NUM_LAYERS, LR, BATCH_SIZE, NUM_NEIGHBORS, EPOCHS, AGGR,
    TEMPORAL_STRATEGY, MAX_STEPS_PER_EPOCH, SEED,
    build_graph_data, build_loaders,
    train_and_eval, count_params, compute_pos_weight_from_task,
)


DATASETS = [("rel-hm", "user-churn")]


def main():
    hb("=== A2 (HGT-only rerun) start ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ds_name, tk_name in DATASETS:
        t0 = time.time()
        data, col_stats_dict, task, _ = build_graph_data(ds_name, tk_name, device, hb=hb)
        hb(f"  data built in {time.time()-t0:.1f}s")
        pos_w = compute_pos_weight_from_task(task)
        hb(f"  pos_weight = {pos_w:.3f}")

        kind = "hgt"
        name = _pretty(kind)
        hb(f"  === model={name} (re-run, no external ReLU) ===")
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
                "note": "external-ReLU removed between HGTConv layers",
            },
            "best_epoch": best["epoch"],
            "val": best["val"],
            "val_best_f1": best["val_best_f1"],
            "wall_seconds": time.time() - t_run,
        })
        hb(f"  {name} best AUC={best['val']['roc_auc']:.4f} "
           f"best-F1={best['val_best_f1']['f1']:.3f}")
    hb("=== A2 (HGT-only rerun) done ===")


if __name__ == "__main__":
    main()
