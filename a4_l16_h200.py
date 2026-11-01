"""L=16 A4 attempt on H200 (141 GB HBM3). Uses the same fan-out and
batch as the previous L=5..8 runs (``[128, 32, 5, 5, ..., 5]``, batch=128)
so the L=16 rows are directly comparable to the earlier depth points.
"""
import a4_l16
a4_l16.BATCH_SIZE = 128


def _matched_neighbors(n_layers: int):
    base = [128, 32]
    return base[:min(2, n_layers)] + [5] * max(0, n_layers - 2)


a4_l16._extreme_neighbors = _matched_neighbors


if __name__ == "__main__":
    a4_l16.main()
