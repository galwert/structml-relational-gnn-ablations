"""Mean-agg rel-stack L=8 skip at batch=32 (previous attempts OOM'd at
batch=64). If this fits it closes the last remaining TBD in the A4
mean-agg skip table."""
import a4_mean
a4_mean.BASELINE_DEPTHS = []
a4_mean.SKIP_DEPTHS     = [8]
a4_mean.TASKS = [("rel-stack", "user-engagement")]
a4_mean.BATCH_SIZE = 32

if __name__ == "__main__":
    a4_mean.main()
