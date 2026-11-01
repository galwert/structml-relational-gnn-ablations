"""A3 multi-seed rerun on the vulnerable claim: rel-hm x HeteroSAGE x
{id, column, llm} spread is 0.0097 AUC at seed=42, which is comparable
to the seed-to-seed std observed in A1 (up to 0.0036). This script runs
seeds 123 and 2024 to check whether the reported ordering
(llm >= column >= id) survives across seeds.

Combined with the base a3_encoders run (seed=42) this gives 3 seeds per
encoder. rel-stack and HGT are not repeated: on rel-stack the encoder
spread is only 0.0012 AUC (safely below any plausible noise), and HGT
rel-hm was already flagged as near-random at base config (footnote in
Table 3).

Base configuration (hidden=128, layers=2, epochs=10, LR=5e-3, no cosine),
identical to a3_encoders -- only the seed varies.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")

from common import setup_paths, log_result, hb
setup_paths()

import torch

from pipeline import (
    CHANNELS, NUM_LAYERS, LR, BATCH_SIZE, NUM_NEIGHBORS, EPOCHS, AGGR,
    TEMPORAL_STRATEGY, MAX_STEPS_PER_EPOCH,
    build_graph_data, build_loaders,
    train_and_eval, count_params, compute_pos_weight_from_task,
)
from a3_encoders import EncoderAblation


DATASETS = [("rel-hm", "user-churn")]
ENCODERS = ["id", "column", "llm"]
SEEDS = [123, 2024]   # seed=42 covered by a3_encoders


def main():
    hb("=== A3-multiseed start ===")
    hb(f"  config: shared base (hidden={CHANNELS}, layers={NUM_LAYERS}, "
       f"epochs={EPOCHS}, LR={LR}); seeds={SEEDS}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ds_name, tk_name in DATASETS:
        t0 = time.time()
        data, col_stats_dict, task, _ = build_graph_data(ds_name, tk_name, device, hb=hb)
        hb(f"  data built in {time.time()-t0:.1f}s")
        pos_w = compute_pos_weight_from_task(task)
        hb(f"  pos_weight = {pos_w:.3f}")
        for seed in SEEDS:
            for enc in ENCODERS:
                hb(f"  === {ds_name} seed={seed} encoder={enc} ===")
                torch.manual_seed(seed)
                if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
                t_run = time.time()
                train_loader, val_loader = build_loaders(data, task)
                model = EncoderAblation(
                    data, col_stats_dict, num_classes=1,
                    target_node=task.entity_table, encoder_kind=enc,
                    channels=CHANNELS, num_layers=NUM_LAYERS, aggr=AGGR,
                ).to(device)
                n_params = count_params(model)
                best = train_and_eval(
                    model, task.entity_table, train_loader, val_loader,
                    device=device, hb=hb, pos_weight=pos_w,
                )
                log_result("a3_multiseed", {
                    "dataset": ds_name, "task": tk_name,
                    "encoder": enc,
                    "backbone": "HeteroSAGE",
                    "seed": seed,
                    "n_params": n_params,
                    "config": {
                        "channels": CHANNELS, "num_layers": NUM_LAYERS,
                        "lr": LR, "batch_size": BATCH_SIZE,
                        "num_neighbors": NUM_NEIGHBORS, "epochs": EPOCHS,
                        "aggr": AGGR, "temporal_strategy": TEMPORAL_STRATEGY,
                        "pos_weight": pos_w,
                        "note": "multi-seed validation of A3 rel-hm HeteroSAGE",
                    },
                    "best_epoch": best["epoch"],
                    "val": best["val"],
                    "val_best_f1": best["val_best_f1"],
                    "wall_seconds": time.time() - t_run,
                })
                hb(f"  {ds_name} seed={seed} encoder={enc} "
                   f"best AUC={best['val']['roc_auc']:.4f}")
                del model, train_loader, val_loader
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        del data, col_stats_dict, task
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    hb("=== A3-multiseed done ===")


if __name__ == "__main__":
    main()
