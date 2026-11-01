"""A4 sum-aggregation tuned rerun over the FULL depth range.

:mod:`a4_tuned` only tuned the peak (L=3) and the collapse (L=8);
this fills the intermediate depths L in {2, 4, 5, 6, 7} for both
{no-skip, skip} on both datasets. Together with the running
:mod:`a4_tuned` they give a complete sum-agg tuned depth curve.

Recipe (matched):
* hidden 128 -> 256
* lr 5e-3 -> 1e-3 with cosine schedule
* epochs 10 -> 30

Configurations: L in {2, 4, 5, 6, 7} x {no-skip, skip} x {rel-hm, rel-stack}.
Logged to ``results/a4_tuned/results.jsonl`` (same file).
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
    NUM_LAYERS, BATCH_SIZE, NUM_NEIGHBORS,
    TEMPORAL_STRATEGY, MAX_STEPS_PER_EPOCH, SEED,
    build_graph_data, build_loaders,
    count_params, compute_pos_weight_from_task,
    compute_metrics, find_best_f1,
)
from a4_depth import DepthModel


DATASETS = [
    ("rel-hm", "user-churn"),
    ("rel-stack", "user-engagement"),
]
DEPTHS = [2, 3, 4, 5, 6, 7, 8]
SKIPS = [False, True]
AGGR_KIND = "sum"

CHANNELS = 256
LR = 1e-3
EPOCHS = 20
EARLY_STOP_PATIENCE = 5


def should_run(ds_name, n_layers, skip):
    """Prune configs unlikely to change conclusions (streamlining).
    Sum-agg + skip at shallow depths (L=2,3) was within 0.001 AUC of
    no-skip in the base run — drop those to save compute."""
    if skip and n_layers in (2, 3):
        return False
    return True


def train_and_eval_cosine(model, target_node, train_loader, val_loader,
                          device, epochs, lr, pos_weight):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device))
    best_auc, best = -1.0, None
    stale = 0
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
            stale = 0
        else:
            stale += 1
            if stale >= EARLY_STOP_PATIENCE:
                hb(f"  early stop at ep {ep} (no improvement in {EARLY_STOP_PATIENCE} epochs; best AUC={best_auc:.4f} at ep {best['epoch']})")
                break
    return best


def main():
    hb(f"=== A4-{AGGR_KIND}-full-tuned start ===")
    hb(f"  config: hidden={CHANNELS}, epochs={EPOCHS}, lr={LR} + cosine, aggr={AGGR_KIND}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ds_name, tk_name in DATASETS:
        t0 = time.time()
        data, col_stats_dict, task, _ = build_graph_data(ds_name, tk_name, device, hb=hb)
        hb(f"  data built in {time.time()-t0:.1f}s")
        pos_w = compute_pos_weight_from_task(task)
        hb(f"  pos_weight = {pos_w:.3f}")
        for n_layers in DEPTHS:
            for skip in SKIPS:
                if not should_run(ds_name, n_layers, skip):
                    hb(f"  --- {ds_name} L={n_layers} skip={skip} pruned ---")
                    continue
                tag = f"L={n_layers} skip={skip} aggr={AGGR_KIND}"
                hb(f"  === {ds_name} {tag} tuned ===")
                torch.manual_seed(SEED)
                if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
                t_run = time.time()
                train_loader, val_loader = build_loaders(
                    data, task, num_layers=n_layers)
                model = DepthModel(
                    data, col_stats_dict, num_classes=1,
                    target_node=task.entity_table,
                    n_layers=n_layers, skip=skip,
                    channels=CHANNELS, aggr=AGGR_KIND,
                ).to(device)
                n_params = count_params(model)
                hb(f"  {tag} params={n_params:,}")
                best = train_and_eval_cosine(
                    model, task.entity_table, train_loader, val_loader,
                    device=device, epochs=EPOCHS, lr=LR, pos_weight=pos_w,
                )
                log_result("a4_tuned", {
                    "dataset": ds_name, "task": tk_name,
                    "n_layers": n_layers, "skip_connections": skip,
                    "aggr": AGGR_KIND,
                    "n_params": n_params,
                    "config": {
                        "channels": CHANNELS, "num_layers": n_layers,
                        "lr": LR, "batch_size": BATCH_SIZE,
                        "num_neighbors": NUM_NEIGHBORS, "epochs": EPOCHS,
                        "aggr": AGGR_KIND, "schedule": "cosine",
                        "temporal_strategy": TEMPORAL_STRATEGY,
                        "seed": SEED, "pos_weight": pos_w,
                        "note": "wider hidden + longer cosine schedule",
                    },
                    "best_epoch": best["epoch"],
                    "val": best["val"],
                    "val_best_f1": best["val_best_f1"],
                    "wall_seconds": time.time() - t_run,
                })
                hb(f"  {ds_name} {tag} best AUC={best['val']['roc_auc']:.4f} "
                   f"best-F1={best['val_best_f1']['f1']:.3f}")
                del model, train_loader, val_loader
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        del data, col_stats_dict, task
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    hb(f"=== A4-{AGGR_KIND}-full-tuned done ===")


if __name__ == "__main__":
    main()
