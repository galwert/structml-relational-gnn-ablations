"""Sum-agg L={2, 7} skip on rel-stack — completes uniform coverage."""
import a4_stack_fits, time, torch
from common import hb
from pipeline import build_graph_data, compute_pos_weight_from_task


def main():
    hb("=== A4 rel-stack skip extra L={2,7} start ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    data, col_stats_dict, task, _ = build_graph_data(
        "rel-stack", "user-engagement", device, hb=hb)
    hb(f"  data built in {time.time()-t0:.1f}s")
    pos_w = compute_pos_weight_from_task(task)
    hb(f"  pos_weight = {pos_w:.3f}")
    for d in [2, 7]:
        a4_stack_fits.run_one(data, col_stats_dict, task, pos_w,
                                    d, True, device)
    hb("=== done ===")


if __name__ == "__main__":
    main()
