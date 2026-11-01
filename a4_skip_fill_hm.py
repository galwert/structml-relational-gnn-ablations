"""Fill in missing skip depths for A4 sum-agg on rel-hm: L={3, 5, 6}."""
import a4_hm_cont
a4_hm_cont.BASELINE_DEPTHS = []
a4_hm_cont.SKIP_DEPTHS     = [3, 5, 6]

if __name__ == "__main__":
    a4_hm_cont.main()
