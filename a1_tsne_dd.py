"""Resume the A1 t-SNE dump for D and DirGNN only — U.npz already exists
from job 68300325. That job died before starting the D loop; we don't know
whether it was OOM or a walltime cut. This script picks up where it left off
and writes D.npz + DirGNN.npz into results/a1_tsne/.
"""
import a1_tsne
a1_tsne.DIRECTIONS = ["D", "DirGNN"]  # U already done

if __name__ == "__main__":
    a1_tsne.main()
