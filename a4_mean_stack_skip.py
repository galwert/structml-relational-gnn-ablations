"""Mean-agg A4 for rel-stack, skip variants L={4, 8}. Splits off for
parallelism. Uses batch=64 (skip=True at L=8 needed 64 for the sum-agg
fits job)."""
import a4_mean
a4_mean.BASELINE_DEPTHS = []
a4_mean.SKIP_DEPTHS     = [4, 8]
a4_mean.TASKS = [("rel-stack", "user-engagement")]
a4_mean.BATCH_SIZE = 64

if __name__ == "__main__":
    a4_mean.main()
