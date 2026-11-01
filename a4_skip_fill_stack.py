"""Fill in missing skip depths for A4 sum-agg on rel-stack: L={3, 5, 6}.
Reuses the batch=64 rel-stack fits config."""
import a4_stack_fits
a4_stack_fits.TASKS = None  # only defined in a4_stack_fits.run_one; overridden by main


def main():
    """Custom entry to run L={3, 5, 6} skip via a4_stack_fits.run_one."""
    import time
    import torch
    from common import hb
    from pipeline import build_graph_data, compute_pos_weight_from_task
    hb("=== A4 rel-stack skip fill L={3,5,6} start ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    data, col_stats_dict, task, _ = build_graph_data(
        "rel-stack", "user-engagement", device, hb=hb)
    hb(f"  data built in {time.time()-t0:.1f}s")
    pos_w = compute_pos_weight_from_task(task)
    hb(f"  pos_weight = {pos_w:.3f}")
    for d in [3, 5, 6]:
        a4_stack_fits.run_one(data, col_stats_dict, task, pos_w,
                                    d, True, device)
    hb("=== done ===")


if __name__ == "__main__":
    main()
