"""Demonstration re-run of rel-hm/user-churn with pos_weight in the BCE loss.

The main ablations logged val AUC ≈ 0.5 across almost every rel-hm variant
because the 82% positive base rate makes the "always predict positive"
shortcut cheap under uniform BCE. This script picks two representative
models — HomoSAGE-U (best rel-hm baseline in A1/A2) and HGT-Hetero (the
type-aware alternative) — and re-runs them with

    pos_weight = n_neg / n_pos   (measured on the train_df)

which up-weights the minority-class ("engaged / not churned") examples in the
loss. Results are logged to results/rel_hm_reweighted/results.jsonl so they
sit alongside the main ablation logs.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")

from common import setup_paths, log_result, hb
setup_paths()

import torch
from src.data import build_data
from src.loaders import (
    build_loaders, make_forward_fn_homo, make_forward_fn_hetero
)
from src.models import HomoSAGE, HGT, count_parameters
from src.train import fit, TrainConfig
from src.features import build_column_features


SHARED = dict(
    hidden=128, n_layers=2, dropout=0.2,
    lr=1e-3, weight_decay=0.0, epochs=15,
    batch_size=2048, neighbors=(15, 10), eval_neighbors=(30, 20), seed=42,
)


def run(model_name, pos_weight):
    hb(f"--- rel-hm/user-churn | model={model_name} | pos_weight={pos_weight:.3f} ---")
    t0 = time.time()
    data = build_data("rel-hm", "user-churn")
    _, feature_dims = build_column_features(data)
    hb(f"  column features: {feature_dims}")
    cfg = TrainConfig(**SHARED)
    tr, va = build_loaders(data, batch_size=cfg.batch_size,
                            train_neighbors=cfg.neighbors,
                            eval_neighbors=cfg.eval_neighbors)
    if model_name == "HomoSAGE":
        m = HomoSAGE(data.hetero, hidden=cfg.hidden, n_layers=cfg.n_layers,
                      dropout=cfg.dropout, directionality="U",
                      target_node=data.target_node,
                      feature_dims=feature_dims)
        fn = make_forward_fn_homo(data.target_node)
    elif model_name == "HGT-Hetero":
        m = HGT(data.hetero, hidden=cfg.hidden, n_layers=cfg.n_layers,
                 dropout=cfg.dropout, target_node=data.target_node,
                 encoder="column", feature_dims=feature_dims)
        fn = make_forward_fn_hetero(data.target_node)
    else:
        raise ValueError(model_name)
    hb(f"  {model_name} params={count_parameters(m):,}")
    metrics = fit(m, tr, va, fn, cfg, pos_weight=pos_weight, verbose=True)
    log_result("rel_hm_reweighted", {
        "dataset": "rel-hm/user-churn",
        "model": model_name,
        "pos_weight": pos_weight,
        "n_params": count_parameters(m),
        "best_epoch": metrics["epoch"], "val": metrics["val"],
        "wall_seconds": time.time() - t0, "config": SHARED,
    })


def main():
    hb("=== rel-hm re-weighted BCE demonstration — start ===")
    # measure pos/neg counts on train
    d = build_data("rel-hm", "user-churn")
    labels = d.train_df[d.label_col].astype(int).values
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    pos_weight = n_neg / max(n_pos, 1)
    hb(f"train label balance: n_pos={n_pos:,}  n_neg={n_neg:,}  pos_weight={pos_weight:.3f}")
    for m in ["HomoSAGE", "HGT-Hetero"]:
        run(m, pos_weight)
    hb("=== rel-hm re-weighted — done ===")


if __name__ == "__main__":
    main()
