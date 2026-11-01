"""A1 multi-seed rerun to validate that the sub-0.007 AUC spread between
U / D / DirGNN is not a single-seed artefact.

Runs 2 additional seeds (123, 2024) x 3 directions x 2 datasets = 12
configs at the shared base configuration (hidden=128, layers=2, LR=5e-3,
10 epochs, no cosine). Identical to the base A1 tables in the report,
only the seed varies. Seed=42 is already covered by ``a1_directionality``, so we do
not repeat it here.

Logs to ``results/a1_multiseed/results.jsonl``. Combined with the
base ``a1_directionality`` numbers (seed=42) this gives us 3 seeds total per config.
If the U > D > DirGNN ordering flips between seeds, or the seed-to-seed
spread is comparable to the between-variant spread, then directionality
really is a nuisance parameter under the tested configurations.
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
from a1_directionality import GNNModel


DATASETS = [
    ("rel-hm", "user-churn"),
    ("rel-stack", "user-engagement"),
]
DIRECTIONS = ["U", "D", "DirGNN"]
SEEDS = [123, 2024]   # seed=42 already covered by a1_directionality (base run)


def main():
    hb("=== A1-multiseed start ===")
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
            for direction in DIRECTIONS:
                hb(f"  === {ds_name} seed={seed} direction={direction} ===")
                torch.manual_seed(seed)
                if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
                t_run = time.time()
                train_loader, val_loader = build_loaders(data, task)
                model = GNNModel(
                    data, col_stats_dict, num_classes=1,
                    target_node=task.entity_table, directionality=direction,
                    channels=CHANNELS, num_layers=NUM_LAYERS, aggr=AGGR,
                ).to(device)
                n_params = count_params(model)
                best = train_and_eval(
                    model, task.entity_table, train_loader, val_loader,
                    device=device, hb=hb, pos_weight=pos_w,
                )
                log_result("a1_multiseed", {
                    "dataset": ds_name, "task": tk_name,
                    "directionality": direction,
                    "seed": seed,
                    "n_params": n_params,
                    "config": {
                        "channels": CHANNELS, "num_layers": NUM_LAYERS,
                        "lr": LR, "batch_size": BATCH_SIZE,
                        "num_neighbors": NUM_NEIGHBORS, "epochs": EPOCHS,
                        "aggr": AGGR, "temporal_strategy": TEMPORAL_STRATEGY,
                        "pos_weight": pos_w,
                        "note": "multi-seed validation of A1 spread",
                    },
                    "best_epoch": best["epoch"],
                    "val": best["val"],
                    "val_best_f1": best["val_best_f1"],
                    "wall_seconds": time.time() - t_run,
                })
                hb(f"  {ds_name} seed={seed} direction={direction} "
                   f"best AUC={best['val']['roc_auc']:.4f}")
                del model, train_loader, val_loader
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        del data, col_stats_dict, task
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    hb("=== A1-multiseed done ===")


if __name__ == "__main__":
    main()
