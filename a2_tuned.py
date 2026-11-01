"""A2 RelBench-pipeline tuned rerun of HeteroSAGE.

A2 already has a tuned HGT-Hetero (:mod:`a2_hgt_tuned`); this fills
in the SAGE-side comparison. If HeteroSAGE stays > HGT-Hetero under a
matched tuning recipe, the base-config SAGE > HGT gap is architectural.

Recipe (matches :mod:`a2_hgt_tuned`):
* hidden 128 -> 256
* lr 5e-3 -> 1e-3 with cosine schedule
* epochs 10 -> 30

Both datasets in one job. Logged to ``results/a2_tuned/results.jsonl``.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")

from common import setup_paths, log_result, hb
setup_paths()

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from pipeline import (
    NUM_LAYERS, BATCH_SIZE, NUM_NEIGHBORS, AGGR,
    TEMPORAL_STRATEGY, MAX_STEPS_PER_EPOCH, SEED,
    build_graph_data, build_loaders,
    count_params, compute_pos_weight_from_task,
    compute_metrics, find_best_f1,
)
from a2_heterogeneity import Backbone  # reuses SAGE/HGT switch


DATASETS = [
    ("rel-hm", "user-churn"),
    ("rel-stack", "user-engagement"),
]
MODELS = ["sage"]  # HGT-Hetero already covered by a2_hgt_tuned

CHANNELS = 256
LR = 1e-3
EPOCHS = 30


def train_and_eval_cosine(model, target_node, train_loader, val_loader,
                          device, epochs, lr, pos_weight):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device))
    best_auc, best = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for i, batch in enumerate(train_loader):
            if i >= MAX_STEPS_PER_EPOCH: break
            batch = batch.to(device)
            opt.zero_grad()
            logits = model(batch).squeeze(-1)
            labels = batch[target_node].y.float()
            loss = criterion(logits, labels)
            loss.backward()
            opt.step()
            total += float(loss); n += 1
        sched.step()
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
        bf = find_best_f1(y_true, y_prob)
        current_lr = sched.get_last_lr()[0]
        hb(f"  ep {ep:>2d}/{epochs} lr={current_lr:.5f} "
           f"loss={total/max(n,1):.4f} AUC={val['roc_auc']:.4f} "
           f"best-F1={bf['f1']:.3f}")
        if val["roc_auc"] > best_auc:
            best_auc = val["roc_auc"]
            best = {"epoch": ep, "val": val, "val_best_f1": bf}
    return best


def _pretty(k):
    return {"sage": "HeteroSAGE-tuned", "hgt": "HGT-Hetero-tuned"}[k]


def main():
    hb("=== A2-tuned start ===")
    hb(f"  config: hidden={CHANNELS}, epochs={EPOCHS}, lr={LR} + cosine")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ds_name, tk_name in DATASETS:
        t0 = time.time()
        data, col_stats_dict, task, _ = build_graph_data(ds_name, tk_name, device, hb=hb)
        hb(f"  data built in {time.time()-t0:.1f}s")
        pos_w = compute_pos_weight_from_task(task)
        hb(f"  pos_weight = {pos_w:.3f}")
        for kind in MODELS:
            name = _pretty(kind)
            hb(f"  === {ds_name} model={name} ===")
            torch.manual_seed(SEED)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
            t_run = time.time()
            train_loader, val_loader = build_loaders(data, task)
            model = Backbone(
                data, col_stats_dict, num_classes=1,
                target_node=task.entity_table, gnn_kind=kind,
                channels=CHANNELS, num_layers=NUM_LAYERS, aggr=AGGR,
            ).to(device)
            n_params = count_params(model)
            hb(f"  {name} params={n_params:,}")
            best = train_and_eval_cosine(
                model, task.entity_table, train_loader, val_loader,
                device=device, epochs=EPOCHS, lr=LR, pos_weight=pos_w,
            )
            log_result("a2_tuned", {
                "dataset": ds_name, "task": tk_name,
                "model": name,
                "n_params": n_params,
                "config": {
                    "channels": CHANNELS, "num_layers": NUM_LAYERS,
                    "lr": LR, "batch_size": BATCH_SIZE,
                    "num_neighbors": NUM_NEIGHBORS, "epochs": EPOCHS,
                    "aggr": AGGR, "schedule": "cosine",
                    "temporal_strategy": TEMPORAL_STRATEGY,
                    "seed": SEED, "pos_weight": pos_w,
                    "note": "wider hidden + longer cosine schedule",
                },
                "best_epoch": best["epoch"],
                "val": best["val"],
                "val_best_f1": best["val_best_f1"],
                "wall_seconds": time.time() - t_run,
            })
            hb(f"  {ds_name} {name} best AUC={best['val']['roc_auc']:.4f} "
               f"best-F1={best['val_best_f1']['f1']:.3f}")
            del model, train_loader, val_loader
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    hb("=== A2-tuned done ===")


if __name__ == "__main__":
    main()
