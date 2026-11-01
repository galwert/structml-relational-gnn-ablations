"""Mean-agg L={2, 7} skip on rel-hm — completes uniform coverage."""
import a4_mean
a4_mean.BASELINE_DEPTHS = []
a4_mean.SKIP_DEPTHS     = [2, 7]
a4_mean.TASKS = [("rel-hm", "user-churn")]

if __name__ == "__main__":
    a4_mean.main()
