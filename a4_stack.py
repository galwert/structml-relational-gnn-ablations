"""A4 on rel-stack/user-engagement — same depth+skip sweep as a4_depth.py,
different dataset. rel-stack is smaller than rel-hm, so the sweep finishes
much faster (a few hours vs many). Appends to ``results/a4_depth/results.jsonl``."""
import a4_depth
a4_depth.DATASETS = [("rel-stack", "user-engagement")]

if __name__ == "__main__":
    a4_depth.main()
