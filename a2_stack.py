"""A2 on rel-stack/user-engagement — same models + pipeline as a2_heterogeneity.py,
different dataset. Runs HeteroSAGE + HGT-Hetero on rel-stack and appends to
``results/a2_heterogeneity/results.jsonl``. Uses the ReLU-free HGT path already patched
in :func:`pipeline.run_gnn`."""
import a2_heterogeneity
a2_heterogeneity.DATASETS = [("rel-stack", "user-engagement")]

if __name__ == "__main__":
    a2_heterogeneity.main()
