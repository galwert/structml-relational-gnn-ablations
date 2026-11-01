"""A1 on rel-stack/user-engagement — same model + pipeline as a1_directionality.py,
different dataset. Runs U/D/DirGNN on rel-stack and appends to
``results/a1_directionality/results.jsonl``."""
import a1_directionality
a1_directionality.DATASETS = [("rel-stack", "user-engagement")]

if __name__ == "__main__":
    a1_directionality.main()
