"""Mean-agg A4 for rel-stack, shallow depths L=2..5 baseline. Splits off
from the rest so it can run in parallel."""
import a4_mean
a4_mean.BASELINE_DEPTHS = [2, 3, 4, 5]
a4_mean.SKIP_DEPTHS     = []
a4_mean.TASKS = [("rel-stack", "user-engagement")]

if __name__ == "__main__":
    a4_mean.main()
