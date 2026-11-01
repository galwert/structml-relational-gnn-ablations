"""Rel-stack A4 continuation — L=5..16 baseline + {4,8,16} skip.

Job 68299779 got L=2/3/4 baseline done (peaked at L=3 = 0.9041) then OOM at
L=5 setup. rel-stack has 7 node types (badges, comments, users, questions,
tags, posts, votes) with lots of edges; the multi-layer sub-batch blew up
memory at 47 GB.

Same fixes as :mod:`a4_hm_deep` — batch=128, tight fan-out
``[128, 32, 5, 5, ...]``, expandable_segments.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
BASELINE_DEPTHS = [5, 6, 7, 8, 16]
SKIP_DEPTHS     = [4, 8, 16]


def _capped_neighbors(n_layers: int):
    base = [128, 32]
    return base[:min(2, n_layers)] + [5] * max(0, n_layers - 2)


def main():
    hb("=== A4 rel-stack deep continuation start ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds_name, tk_name = "rel-stack", "user-engagement"
    t0 = time.time()
    data, col_stats_dict, task, _ = build_graph_data(ds_name, tk_name, device, hb=hb)
    hb(f"  data built in {time.time()-t0:.1f}s")
    pos_w = compute_pos_weight_from_task(task)
    hb(f"  pos_weight = {pos_w:.3f}")

    combos = ([(d, False) for d in BASELINE_DEPTHS] +
              [(d, True)  for d in SKIP_DEPTHS])
    for n_layers, skip in combos:
        tag = f"L={n_layers} skip={skip}"
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
                "note": "batch=128, tight fan-out [128,32,5,5,...], expandable_segments",
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
    hb("=== A4 rel-stack deep continuation done ===")


if __name__ == "__main__":
    main()
