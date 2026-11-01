"""HGT-Hetero tuning experiment: test whether the SAGE-vs-HGT gap
observed in A2 is real or a hyperparameter artefact.

Changes vs the base A2 run (a2_heterogeneity.py, HGT-Hetero):
* hidden: 128 -> 256
* heads: 4 -> 8 (giving 32-d per head with the wider hidden)
* epochs: 10 -> 30
* lr: 5e-3 -> 1e-3 with cosine-decay schedule (0 warmup, 30-epoch cycle)
* batch size stays at 512

Rationale: HGT's original paper recommends wider hidden + more heads +
longer training than SAGE's default RelBench configuration. If the
SAGE-vs-HGT gap survives this recipe, we can say with confidence that
SAGE dominates at this data scale; otherwise the base-config gap was
under-tuning, not architectural.

Results log to ``results/a2_hgt_tuned/results.jsonl``.
"""
import os, sys, time, warnings, math
warnings.filterwarnings("ignore")

from common import setup_paths, log_result, hb
setup_paths()

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from torch.optim.lr_scheduler import CosineAnnealingLR

from pipeline import (
    build_graph_data, build_loaders, run_gnn,
    count_params, compute_pos_weight_from_task,
    compute_metrics, find_best_f1,
    TEMPORAL_STRATEGY, SEED, MAX_STEPS_PER_EPOCH,
)
from relbench.modeling.nn import HeteroEncoder, HeteroTemporalEncoder


DATASETS = [
    ("rel-hm", "user-churn"),
    ("rel-stack", "user-engagement"),
]

# Tuned hyperparameters
CHANNELS = 256          # was 128
NUM_HEADS = 8           # was 4
NUM_LAYERS = 2
LR = 1e-3               # was 5e-3
EPOCHS = 30             # was 10
BATCH_SIZE = 512


class HGTTuned(nn.Module):
    def __init__(self, data, col_stats_dict, num_classes, target_node,
                 channels=CHANNELS, num_layers=NUM_LAYERS,
                 num_heads=NUM_HEADS):
        super().__init__()
        self.target_node = target_node
        self.encoder = HeteroEncoder(
            channels=channels,
            node_to_col_names_dict={
                nt: data[nt].tf.col_names_dict for nt in data.node_types
            },
            node_to_col_stats=col_stats_dict,
        )
        self.temporal_encoder = HeteroTemporalEncoder(
            node_types=[nt for nt in data.node_types if "time" in data[nt]],
            channels=channels,
        )
        metadata = (list(data.node_types), list(data.edge_types))
        self.gnn = nn.ModuleList([
            HGTConv(channels, channels, metadata, heads=num_heads)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(channels, num_classes)

    def forward(self, batch: HeteroData) -> torch.Tensor:
        x_dict = self.encoder(batch.tf_dict)
        rel = self.temporal_encoder(
            batch[self.target_node].seed_time,
            batch.time_dict, batch.batch_dict)
        for nt, r in rel.items():
            x_dict[nt] = x_dict[nt] + r
        out = run_gnn(self.gnn, "hgt", x_dict, batch.edge_index_dict)
        bsz = batch[self.target_node].batch_size
        return self.head(out[self.target_node][:bsz])


def train_and_eval_cosine(model, target_node, train_loader, val_loader,
                           device, epochs, lr, pos_weight):
    """Same as train_and_eval but with cosine LR schedule + longer epoch
    budget."""
    import numpy as np
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


def main():
    hb("=== A2-HGT-tuned start ===")
    hb(f"  config: hidden={CHANNELS}, heads={NUM_HEADS}, epochs={EPOCHS}, "
       f"lr={LR} + cosine")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ds_name, tk_name in DATASETS:
        t0 = time.time()
        data, col_stats_dict, task, _ = build_graph_data(ds_name, tk_name, device, hb=hb)
        hb(f"  data built in {time.time()-t0:.1f}s")
        pos_w = compute_pos_weight_from_task(task)
        hb(f"  pos_weight = {pos_w:.3f}")

        hb(f"  === HGT-Hetero-tuned on {ds_name} ===")
        torch.manual_seed(SEED)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
        t_run = time.time()
        train_loader, val_loader = build_loaders(
            data, task, batch_size=BATCH_SIZE)
        model = HGTTuned(
            data, col_stats_dict, num_classes=1,
            target_node=task.entity_table,
        ).to(device)
        n_params = count_params(model)
        hb(f"  HGT-Hetero-tuned params={n_params:,}")
        best = train_and_eval_cosine(
            model, task.entity_table, train_loader, val_loader,
            device, epochs=EPOCHS, lr=LR, pos_weight=pos_w,
        )
        log_result("a2_hgt_tuned", {
            "dataset": ds_name, "task": tk_name,
            "model": "HGT-Hetero-tuned",
            "n_params": n_params,
            "config": {
                "channels": CHANNELS, "num_layers": NUM_LAYERS,
                "num_heads": NUM_HEADS,
                "lr": LR, "batch_size": BATCH_SIZE,
                "epochs": EPOCHS, "schedule": "cosine",
                "temporal_strategy": TEMPORAL_STRATEGY,
                "seed": SEED, "pos_weight": pos_w,
                "note": "wider hidden + more heads + longer schedule",
            },
            "best_epoch": best["epoch"],
            "val": best["val"],
            "val_best_f1": best["val_best_f1"],
            "wall_seconds": time.time() - t_run,
        })
        hb(f"  HGT-Hetero-tuned best AUC={best['val']['roc_auc']:.4f} "
           f"best-F1={best['val_best_f1']['f1']:.3f}")
        del model, train_loader, val_loader
        torch.cuda.empty_cache()
    hb("=== A2-HGT-tuned done ===")


if __name__ == "__main__":
    main()
