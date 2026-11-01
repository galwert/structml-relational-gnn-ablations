"""A3 on rel-stack/user-engagement — same encoder ablation as a3_encoders.py,
different dataset. Runs id / column / llm encoders on rel-stack and appends
to ``results/a3_encoders/results.jsonl``."""
import a3_encoders
a3_encoders.DATASETS = [("rel-stack", "user-engagement")]

if __name__ == "__main__":
    a3_encoders.main()
