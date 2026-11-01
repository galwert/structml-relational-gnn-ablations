"""A3 RelBench-pipeline tuned rerun of all six encoder x backbone combinations.

Tests whether the base-config encoder ordering (id / column / llm) or the
HGT-rel-hm collapse (where column and llm collapse to ~0.55 while id
survives at ~0.62) survives a proper tuning budget.

Recipe (matches :mod:`a2_hgt_tuned` and :mod:`a1_tuned`):
* hidden 128 -> 256
* HGT heads 4 -> 8
* lr 5e-3 -> 1e-3 with cosine schedule
* epochs 10 -> 30

Both backbones (HeteroSAGE, HGT-Hetero) x three encoders (id, column,
llm) x two datasets = 12 runs. Logged to
``results/a3_tuned/results.jsonl``.
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
from a3_encoders import EncoderAblation           # HeteroSAGE backbone
from a3_hgt import EncoderAblationHGT    # HGT backbone


DATASETS = [
    ("rel-hm", "user-churn"),
    ("rel-stack", "user-engagement"),
]
ENCODERS = ["id", "column", "llm"]
BACKBONES = ["sage", "hgt"]

CHANNELS = 256
NUM_HEADS = 8
LR = 1e-3
EPOCHS = 20
EARLY_STOP_PATIENCE = 5   # break if best AUC hasn't improved in N epochs


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


def build_model(backbone, data, col_stats_dict, target_node, encoder_kind):
    if backbone == "sage":
        return EncoderAblation(
            data, col_stats_dict, num_classes=1,
            target_node=target_node, encoder_kind=encoder_kind,
            channels=CHANNELS, num_layers=NUM_LAYERS, aggr=AGGR,
        )
    else:  # hgt
        return EncoderAblationHGT(
            data, col_stats_dict, num_classes=1,
            target_node=target_node, encoder_kind=encoder_kind,
            channels=CHANNELS, num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
        )


def _pretty_backbone(b): return {"sage": "HeteroSAGE", "hgt": "HGT-Hetero"}[b]


def main():
    hb("=== A3-tuned start ===")
    hb(f"  config: hidden={CHANNELS}, heads={NUM_HEADS}, epochs={EPOCHS}, lr={LR} + cosine")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ds_name, tk_name in DATASETS:
        t0 = time.time()
        data, col_stats_dict, task, _ = build_graph_data(ds_name, tk_name, device, hb=hb)
        hb(f"  data built in {time.time()-t0:.1f}s")
        pos_w = compute_pos_weight_from_task(task)
        hb(f"  pos_weight = {pos_w:.3f}")
        for backbone in BACKBONES:
            for enc in ENCODERS:
                bb = _pretty_backbone(backbone)
                hb(f"  === {ds_name} {bb} encoder={enc} ===")
                torch.cuda.empty_cache()
                torch.manual_seed(SEED)
                if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
                t_run = time.time()
                train_loader, val_loader = build_loaders(data, task)
                model = build_model(
                    backbone, data, col_stats_dict, task.entity_table, enc
                ).to(device)
                n_params = count_params(model)
                hb(f"  {bb}/{enc}-tuned params={n_params:,}")
                best = train_and_eval_cosine(
                    model, task.entity_table, train_loader, val_loader,
                    device=device, epochs=EPOCHS, lr=LR, pos_weight=pos_w,
                )
                log_result("a3_tuned", {
                    "dataset": ds_name, "task": tk_name,
                    "encoder": enc,
                    "backbone": bb,
                    "n_params": n_params,
                    "config": {
                        "channels": CHANNELS, "num_layers": NUM_LAYERS,
                        "num_heads": NUM_HEADS if backbone == "hgt" else None,
                        "lr": LR, "batch_size": BATCH_SIZE,
                        "num_neighbors": NUM_NEIGHBORS, "epochs": EPOCHS,
                        "aggr": AGGR, "schedule": "cosine",
                        "temporal_strategy": TEMPORAL_STRATEGY,
                        "seed": SEED, "pos_weight": pos_w,
                        "note": "wider hidden + more heads + longer cosine schedule",
                    },
                    "best_epoch": best["epoch"],
                    "val": best["val"],
                    "val_best_f1": best["val_best_f1"],
                    "wall_seconds": time.time() - t_run,
                })
                hb(f"  {ds_name} {bb}/{enc}-tuned best AUC={best['val']['roc_auc']:.4f} "
                   f"best-F1={best['val_best_f1']['f1']:.3f}")
                del model, train_loader, val_loader
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        del data, col_stats_dict, task
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    hb("=== A3-tuned done ===")


if __name__ == "__main__":
    main()
