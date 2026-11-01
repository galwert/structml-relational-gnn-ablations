# Experiments

All experiments share `pipeline.py` (graph construction via `make_pkey_fkey_graph`,
encoders, HeteroGraphSAGE / HGT model building blocks, training loop, metrics
incl. MCC / BA / best-F1 threshold grid) and `common.py` (`setup_paths`,
`log_result`, heartbeat logging). Each script appends one JSON line per run to
`results/<aspect>/results.jsonl`.

## File map

| Aspect | Main script | Variants |
|---|---|---|
| A1 directionality | `a1_directionality.py` (U / D / DirGNN) | `a1_stack.py` (rel-stack only), `a1_tuned.py`, `a1_multiseed.py` + `a1_multiseed_tail.py` (seeds 42/123/2024), `a1_tsne*.py` (embedding projections) |
| A2 heterogeneity | `a2_heterogeneity.py` (HeteroSAGE) | `a2_hgt.py`, `a2_hgt_homo.py`, `a2_homo_sage.py`, `a2_stack.py`, `*_tuned.py` per variant |
| A3 encoders | `a3_encoders.py` (id / column / llm, HeteroSAGE) | `a3_hgt.py` (spec backbone), `a3_stack.py`, `a3_tuned.py` (all 12 configs), `a3_multiseed.py` |
| A4 depth | `a4_depth.py` (sum agg, L sweep) | `a4_mean.py` (mean agg + per-layer MAD), `a4_*skip*.py` / `a4_*fits*.py` / `a4_missing.py` / `a4_hm_*.py` / `a4_stack_*.py` (memory-constrained fill-in jobs), `a4_l16*.py` (OOM'd — see report), `a4_tuned.py`, `a4_sum_tuned.py`, `a4_mean_tuned.py` |
| Sanity | `rel_hm_reweighted.py` (class-reweighting probe, uses `src/`) | |

## Renamed after submission

Files were renamed post-submission for readability (the `_rdl` tag, short for
relational deep learning, was dropped; superseded prototype scripts were removed).
The code content and all results are unchanged. Mapping from the submitted names:

| Submitted | Now | Submitted | Now |
|---|---|---|---|
| `rdl_common.py` | `pipeline.py` | `a3_rdl.py` | `a3_encoders.py` |
| `a1_rdl.py` | `a1_directionality.py` | `a3_rdl_hgt.py` | `a3_hgt.py` |
| `a1_rdl_stack.py` | `a1_stack.py` | `a3_rdl_stack.py` | `a3_stack.py` |
| `a1_rdl_tuned.py` | `a1_tuned.py` | `a3_rdl_tuned.py` | `a3_tuned.py` |
| `a1_rdl_multiseed.py` | `a1_multiseed.py` | `a3_rdl_multiseed.py` | `a3_multiseed.py` |
| `a1_rdl_multiseed_tail.py` | `a1_multiseed_tail.py` | `a4_rdl.py` | `a4_depth.py` |
| `a2_rdl.py` | `a2_heterogeneity.py` | `a4_rdl_mean.py` | `a4_mean.py` |
| `a2_rdl_hgt.py` | `a2_hgt.py` | `a4_rdl_tuned.py` | `a4_tuned.py` |
| `a2_rdl_hgt_tuned.py` | `a2_hgt_tuned.py` | `a4_rdl_mean_tuned.py` | `a4_mean_tuned.py` |
| `a2_rdl_hgt_homo.py` | `a2_hgt_homo.py` | `a4_rdl_sum_full_tuned.py` | `a4_sum_tuned.py` |
| `a2_rdl_hgt_homo_tuned.py` | `a2_hgt_homo_tuned.py` | `a4_rdl_fits.py` | `a4_fits.py` |
| `a2_rdl_homo_sage.py` | `a2_homo_sage.py` | `a4_rdl_hm_cont.py` | `a4_hm_cont.py` |
| `a2_rdl_homo_sage_tuned.py` | `a2_homo_sage_tuned.py` | `a4_rdl_hm_deep.py` | `a4_hm_deep.py` |
| `a2_rdl_stack.py` | `a2_stack.py` | `a4_rdl_l16.py` / `_h200` | `a4_l16.py` / `_h200` |
| `a2_rdl_tuned.py` | `a2_tuned.py` | `a4_rdl_missing.py` | `a4_missing.py` |
| | | `a4_rdl_stack*.py` | `a4_stack*.py` |

Result directories were renamed to match (`results/a1_rdl` → `results/a1_directionality`,
`results/rdl_cache` → `results/graph_cache`, etc.); the notebook loads from the new
names. Removed as superseded prototypes (still present in the submitted
`experiments.zip`): `a1_directionality.py` (old version), `a1_paper.py`,
`a2_heterogeneity.py` (old version), `a3_node_features.py`, `a3_sample50k.py`,
`a4_oversmoothing.py`, `a4_depth_fill.py`, the old `a{1..4}_tuned.py`,
`smoke_test.py`, and `src/{diagnostics,llm_encoding,report_assembly}.py`.
Raw run logs under `results/` keep the original names they were produced with.
