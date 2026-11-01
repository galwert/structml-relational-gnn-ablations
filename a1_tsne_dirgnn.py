"""Run just the DirGNN variant of :mod:`a1_tsne`. U.npz and D.npz already
exist from the previous partial runs (job 68300325 saved U, job 68300458
saved D before crashing on a bincount log-line bug in a1_tsne.main). This
picks up the last remaining variant.
"""
import a1_tsne
a1_tsne.DIRECTIONS = ["DirGNN"]

if __name__ == "__main__":
    a1_tsne.main()
