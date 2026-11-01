"""Retry sum-agg rel-hm L={6, 7} skip with the memory-tight deep-mode
config (batch=128, fan-out ``[128, 32, 5, 5, ...]``). The previous
attempts (68308135 for L=6, 68308150 for L=7) OOM'd at batch=512 with
the wider fan-out."""
import a4_hm_deep
a4_hm_deep.BASELINE_DEPTHS = []
a4_hm_deep.SKIP_DEPTHS     = [6, 7]

if __name__ == "__main__":
    a4_hm_deep.main()
