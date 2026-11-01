"""Fill in missing skip depths for A4 mean-agg on rel-hm: L={3, 5, 6}.
Splits from the parallel a4_mean_hm_rest job."""
import a4_mean
a4_mean.BASELINE_DEPTHS = []
a4_mean.SKIP_DEPTHS     = [3, 5, 6]
a4_mean.TASKS = [("rel-hm", "user-churn")]

if __name__ == "__main__":
    a4_mean.main()
