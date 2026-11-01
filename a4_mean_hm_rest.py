"""Mean-agg A4 finisher for rel-hm — L=8 baseline + L={4,8} skip.
rel-hm L=2..7 baseline already landed via the cancelled 68307066 run.
Splits off the rest so it can run in parallel with the rel-stack jobs."""
import a4_mean
a4_mean.BASELINE_DEPTHS = [8]
a4_mean.SKIP_DEPTHS     = [4, 8]
a4_mean.TASKS = [("rel-hm", "user-churn")]

if __name__ == "__main__":
    a4_mean.main()
