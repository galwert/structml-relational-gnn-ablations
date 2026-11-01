"""Train each A1 variant (U / D / DirGNN) on rel-stack briefly with the RelBench
pipeline, then dump the target-node embeddings (the layer directly before the
classifier head) and their labels to ``results/a1_tsne/{dir}.npz``.

The notebook's t-SNE cell reads those files and visualises the three learned
embedding spaces side-by-side, coloured by the label.

We use rel-stack instead of rel-hm because:
* rel-stack's val split is ~2 orders of magnitude smaller — quicker to embed.
* the class distribution is more informative for a 2-class colour split
  (rel-hm is 82% positive, so t-SNE panels are dominated by one colour).

Only 5 000 val target nodes are sampled per variant — enough to see cluster
structure, cheap enough to fit each fit+embed+dump into a few minutes on an
L40.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")

from common import setup_paths, log_result, hb
setup_paths()

import numpy as np
import torch

from pipeline import (
    CHANNELS, NUM_LAYERS, LR, BATCH_SIZE, EPOCHS, AGGR, MAX_STEPS_PER_EPOCH,
    SEED, build_graph_data, build_loaders, safe_hetero_forward,
    count_params, compute_pos_weight_from_task,
    compute_metrics,
)
from a1_directionality import GNNModel


DATASET, TASK = "rel-stack", "user-engagement"
DIRECTIONS = ["U", "D", "DirGNN"]
N_EMB_SAMPLES = 5_000
OUT_DIR = "/home/wertheizer/structeredML/project/results/a1_tsne"
os.makedirs(OUT_DIR, exist_ok=True)


@torch.no_grad()
def dump_embeddings(model: GNNModel, val_loader, target_node: str,
                    device: str, max_samples: int):
    """Run the model on val batches and record (embedding, label) tuples for
    the target-node seeds. Embedding = the target-node representation right
    before the classifier head (post-GNN, so *learned*)."""
    model.eval()
    ems, ys = [], []
    total = 0
    for batch in val_loader:
        batch = batch.to(device)
        # Manually re-run forward up to the last hidden — mimic GNNModel.forward
        x_dict = model.encoder(batch.tf_dict)
        rel_time_dict = model.temporal_encoder(
            batch[target_node].seed_time,
            batch.time_dict, batch.batch_dict,
        )
        for nt, rel in rel_time_dict.items():
            x_dict[nt] = x_dict[nt] + rel
        if model.directionality == "U":
            out = safe_hetero_forward(model.gnn, x_dict, batch.edge_index_dict)
        elif model.directionality == "D":
            ei_fwd = {et: ei for et, ei in batch.edge_index_dict.items()
                      if not et[1].startswith("rev_")}
            out = safe_hetero_forward(model.gnn, x_dict, ei_fwd)
        else:  # DirGNN — same code path as GNNModel.forward
            import torch.nn.functional as F
            ei_fwd = {et: ei for et, ei in batch.edge_index_dict.items()
                      if not et[1].startswith("rev_")}
            ei_bwd = {et: ei for et, ei in batch.edge_index_dict.items()
                      if et[1].startswith("rev_")}
            out_fwd = safe_hetero_forward(model.gnn, x_dict, ei_fwd)
            out_bwd = safe_hetero_forward(model.gnn_bwd, x_dict, ei_bwd)
            out = {}
            for nt in x_dict:
                if nt in out_fwd and nt in out_bwd:
                    out[nt] = F.relu(model.merge(
                        torch.cat([out_fwd[nt], out_bwd[nt]], dim=-1)))
                elif nt in out_fwd:
                    out[nt] = out_fwd[nt]
                else:
                    out[nt] = out_bwd[nt]

        bsz = batch[target_node].batch_size
        emb = out[target_node][:bsz].detach().cpu().numpy()
        y = batch[target_node].y[:bsz].detach().cpu().numpy()
        ems.append(emb)
        ys.append(y)
        total += bsz
        if total >= max_samples:
            break
    Z = np.concatenate(ems, axis=0)[:max_samples]
    Y = np.concatenate(ys, axis=0)[:max_samples]
    return Z, Y


def main():
    hb("=== A1-tSNE start ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    data, col_stats_dict, task, _ = build_graph_data(DATASET, TASK, device, hb=hb)
    hb(f"  data built in {time.time()-t0:.1f}s")
    pos_w = compute_pos_weight_from_task(task)
    hb(f"  pos_weight = {pos_w:.3f}")

    for direction in DIRECTIONS:
        hb(f"  === direction={direction} ===")
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        t_run = time.time()

        train_loader, val_loader = build_loaders(data, task)
        model = GNNModel(
            data, col_stats_dict, num_classes=1,
            target_node=task.entity_table, directionality=direction,
        ).to(device)
        hb(f"  {direction} params={count_params(model):,}")

        # Manual fit loop — reuse train_and_eval-style code but keep the model
        # in scope so we can dump embeddings afterwards.
        from torch.nn import BCEWithLogitsLoss
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=device))
        for ep in range(1, EPOCHS + 1):
            model.train()
            total, n = 0.0, 0
            for i, batch in enumerate(train_loader):
                if i >= MAX_STEPS_PER_EPOCH: break
                batch = batch.to(device)
                opt.zero_grad()
                logits = model(batch).squeeze(-1)
                labels = batch[task.entity_table].y.float()
                loss = criterion(logits, labels)
                loss.backward()
                opt.step()
                total += float(loss); n += 1
            hb(f"  ep {ep:>2d}/{EPOCHS}  loss={total/max(n,1):.4f}")

        Z, Y = dump_embeddings(model, val_loader, task.entity_table,
                                device, max_samples=N_EMB_SAMPLES)
        out_path = os.path.join(OUT_DIR, f"{direction}.npz")
        np.savez(out_path, Z=Z, Y=Y, direction=direction, dataset=DATASET)
        hb(f"  -> saved {out_path} (Z shape={Z.shape}, "
           f"class dist={np.bincount(Y.astype(np.int64))})")

        # also record the training time in _progress.log for consistency
        hb(f"  direction={direction} done in {time.time()-t_run:.1f}s")
    hb("=== A1-tSNE done ===")


if __name__ == "__main__":
    main()
