"""Fill in the missing A4 depths on both datasets:

* rel-hm: L=16 baseline + L={4, 8, 16} skip
* rel-stack: L=8, L=16 baseline + L={4, 8, 16} skip

Jobs 68300599/68300600 OOM'd at the deepest layers because I set
``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` too late (inside the
Python script, after torch had already initialised CUDA). The sbatch now
exports it before Python launches. Combined with the same tight fan-out
``[128, 32, 5, 5, ...]`` and ``batch=128`` this run should finally close the
depth sweep.

Both datasets in one job so we only pay the make_pkey_fkey_graph cost once
per dataset (cached, so ~30 s each).
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")

from common import setup_paths, log_result, hb
setup_paths()

import torch
from a4_depth import DepthModel
from pipeline import (
    CHANNELS, LR, EPOCHS, AGGR, TEMPORAL_STRATEGY,
    MAX_STEPS_PER_EPOCH, SEED,
    build_graph_data, train_and_eval, count_params, compute_pos_weight_from_task,
)


BATCH_SIZE = 128
# (dataset, task, baseline_depths_missing, skip_depths_missing)
TASKS = [
    ("rel-hm",    "user-churn",       [16],          [4, 8, 16]),
    ("rel-stack", "user-engagement", [8, 16],        [4, 8, 16]),
]


def _capped_neighbors(n_layers: int):
    base = [128, 32]
    return base[:min(2, n_layers)] + [5] * max(0, n_layers - 2)


def run_one(data, col_stats_dict, task, ds_name, tk_name,
             pos_w, n_layers, skip, device):
    tag = f"{ds_name} L={n_layers} skip={skip}"
    hb(f"  === {tag} ===")
    torch.cuda.empty_cache()
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    t_run = time.time()

    from relbench.modeling.graph import get_node_train_table_input
    from torch_geometric.loader import NeighborLoader
    neighbors = _capped_neighbors(n_layers)
    hb(f"  fan-out per layer = {neighbors}, batch={BATCH_SIZE}")
    train_ti = get_node_train_table_input(
        table=task.get_table("train"), task=task)
    val_ti = get_node_train_table_input(
        table=task.get_table("val"), task=task)
    train_loader = NeighborLoader(
        data, num_neighbors=neighbors, time_attr="time",
        input_nodes=train_ti.nodes, input_time=train_ti.time,
        transform=train_ti.transform, batch_size=BATCH_SIZE,
        temporal_strategy=TEMPORAL_STRATEGY, shuffle=True, num_workers=0,
    )
    val_loader = NeighborLoader(
        data, num_neighbors=neighbors, time_attr="time",
        input_nodes=val_ti.nodes, input_time=val_ti.time,
        transform=val_ti.transform, batch_size=BATCH_SIZE,
        temporal_strategy=TEMPORAL_STRATEGY, shuffle=False, num_workers=0,
    )

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
            "num_neighbors": neighbors,
            "epochs": EPOCHS,
            "aggr": AGGR, "temporal_strategy": TEMPORAL_STRATEGY,
            "seed": SEED, "pos_weight": pos_w,
            "note": "batch=128, tight fan-out, expandable_segments=True in env",
        },
        "best_epoch": best["epoch"],
        "val": best["val"],
        "val_best_f1": best["val_best_f1"],
        "wall_seconds": time.time() - t_run,
    })
    hb(f"  {tag} best AUC={best['val']['roc_auc']:.4f} "
       f"best-F1={best['val_best_f1']['f1']:.3f}")
    del model, train_loader, val_loader
    torch.cuda.empty_cache()


def main():
    hb("=== A4 missing-depths run start ===")
    hb(f"  PYTORCH_CUDA_ALLOC_CONF = {os.environ.get('PYTORCH_CUDA_ALLOC_CONF','<unset>')}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ds_name, tk_name, baseline_depths, skip_depths in TASKS:
        t0 = time.time()
        data, col_stats_dict, task, _ = build_graph_data(ds_name, tk_name, device, hb=hb)
        hb(f"  {ds_name} data built in {time.time()-t0:.1f}s")
        pos_w = compute_pos_weight_from_task(task)
        hb(f"  pos_weight = {pos_w:.3f}")
        for d in baseline_depths:
            run_one(data, col_stats_dict, task, ds_name, tk_name,
                    pos_w, d, False, device)
        for d in skip_depths:
            run_one(data, col_stats_dict, task, ds_name, tk_name,
                    pos_w, d, True, device)
        # release graph before switching datasets
        del data, col_stats_dict, task
        torch.cuda.empty_cache()
    hb("=== A4 missing-depths run done ===")


if __name__ == "__main__":
    main()
