"""Fill in missing skip depths for A4 mean-agg on rel-stack: L={3, 5, 6}."""
import a4_mean
a4_mean.BASELINE_DEPTHS = []
a4_mean.SKIP_DEPTHS     = [3, 5, 6]
a4_mean.TASKS = [("rel-stack", "user-engagement")]
a4_mean.BATCH_SIZE = 64

if __name__ == "__main__":
    a4_mean.main()
