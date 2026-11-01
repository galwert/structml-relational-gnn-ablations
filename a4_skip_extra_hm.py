"""Sum-agg L={2, 7} skip on rel-hm — completes uniform coverage."""
import a4_hm_cont
a4_hm_cont.BASELINE_DEPTHS = []
a4_hm_cont.SKIP_DEPTHS     = [2, 7]

if __name__ == "__main__":
    a4_hm_cont.main()
