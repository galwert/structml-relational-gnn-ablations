"""Mean-agg A4 for rel-stack, deep baselines L=6..8. L=8 needs the batch=32
config for memory (same fix as the sum-agg run). Splits off for parallelism."""
import a4_mean
a4_mean.BASELINE_DEPTHS = [6, 7, 8]
a4_mean.SKIP_DEPTHS     = []
a4_mean.TASKS = [("rel-stack", "user-engagement")]
# L=8 rel-stack was OOM at batch=128 in the sum-agg fits job — the mean-agg
# path allocates similarly, so drop to batch=32 for the whole rel-stack deep
# sweep to be safe (small runtime penalty at L=6, big memory saving at L=8).
a4_mean.BATCH_SIZE = 32

if __name__ == "__main__":
    a4_mean.main()
