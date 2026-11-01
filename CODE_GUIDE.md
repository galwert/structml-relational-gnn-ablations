# Code guide — what each file contains and where to find everything

Companion to `README.md` (which has the per-aspect file map and the
post-submission rename table). This document explains **what each file does,
function by function**, and ends with a **"where is X implemented?" lookup
table** for fast navigation.

## Architecture in one paragraph

Three layers. **Layer 1** is two central modules every experiment imports:
`pipeline.py` (data building, encoders, GNN construction, training loop,
metrics) and `common.py` (path setup, JSONL result logging, heartbeat
logging). **Layer 2** is one "main" script per aspect that defines its model
class on top of the pipeline and sweeps the studied axis. **Layer 3** is thin
wrapper scripts that import a layer-2 module, override a few module-level
constants (dataset list, depths, seeds), and call its `main()` — these exist
because cluster jobs hit SLURM timeouts and GPU OOMs and had to be resumed in
pieces. Every run appends one JSON line (config + all metrics) to
`results/<aspect>/results.jsonl`; the notebook renders all tables and figures
from those files offline.

---

## Layer 1 — central modules

### `pipeline.py` (~400 lines) — the shared pipeline

| Lines | Symbol | What it is |
|---|---|---|
| 45–54 | `CHANNELS=128, NUM_LAYERS=2, LR=5e-3, BATCH_SIZE=512, NUM_NEIGHBORS=128, EPOCHS=10, AGGR="sum", SEED=42` | **The shared base configuration** — the fair-comparison constants every aspect holds fixed |
| 61–85 | `SentenceEmbedder` | MiniLM sentence-transformer wrapper for text columns. **NaN-tolerant**: maps `None`/`NaN` cells to `""` before encoding (lines 69–76) |
| 92–111 | `build_graph_data()` | **Graph construction**: `relbench.get_dataset/get_task` → `get_stype_proposal` → `make_pkey_fkey_graph` with the text-embedder config. **Disk-cached** under `results/graph_cache/<dataset>` (line 104) — first build ~7 min, then ~10 s |
| 114–139 | `build_loaders()` | Train/val `NeighborLoader`s with temporal sampling. **Fan-out halves per layer**: `num_neighbors / 2**i` (line 122) |
| 157–169 | `safe_hetero_forward()` | Robust HeteroGraphSAGE forward. Two fixes: (1) drops edge types with 0 edges in the sampled subgraph (crash in SAGEConv aggregation otherwise), (2) **carry-forward** — a node type that receives no messages keeps its previous representation (lines 164–166). Without this the D-only ablation crashes on isolated node types |
| 176–196 | `IdEncoder` | A3 "id": one learnable 128-d vector **per node type** (not per node — a per-node table on rel-hm's 1.4M customers would add 180M params and break the capacity comparison; see docstring) |
| 199–221 | `ColumnEncoder` | A3 "column": `HeteroEncoder` with **text stypes filtered out** (line 210–211) |
| 224–239 | `LLMEncoder` | A3 "llm": full `HeteroEncoder` including `text_embedded` columns (pre-embedded at data-build time) |
| 245–267 | `build_gnn()` | Backbone factory: `'sage'` → RelBench `HeteroGraphSAGE`; `'hgt'` → stack of `HGTConv` with the graph's metadata |
| 270–293 | `run_gnn()` | Forward for either backbone. **HGT subtlety** (docstring, lines 273–280): no external ReLU between HGTConv layers — adding one makes HGT collapse to constant 0.5 |
| 299–309 | `compute_metrics()` | AUC, AUPRC, precision, recall, accuracy, F1 at a threshold |
| 312–328 | `find_best_f1()` | **The F1\* threshold grid**: 91 thresholds in [0.05, 0.95], returns best F1 + its threshold/precision/recall |
| 331–338 | `compute_pos_weight_from_task()` | **pos_weight = n_neg / n_pos** from the train table (≈0.22 on rel-hm, ≈18 on rel-stack) |
| 344–397 | `train_and_eval()` | **The training loop**: Adam + `BCEWithLogitsLoss(pos_weight)`, per-epoch val sweep, keeps the best-AUC epoch. Model-agnostic — expects `model(batch) -> logits` |
| 400–401 | `count_params()` | trainable-parameter count reported in the tables |

Note: MCC and BA are **not** computed here — they're derived in the notebook:
`notebook.py :: mcc_ba_from_pr()` reconstructs the confusion matrix in
unit-normalized proportions from the logged (precision, recall) + the dataset
base rate, and `explode_val()` adds `MCC`/`BA` columns to every results table.
This design meant old result files never needed re-running when the extra
metrics were added.

### `common.py` (~50 lines) — run boilerplate

- `setup_paths()` — inserts project root into `sys.path`, `chdir`s to root so scripts run from anywhere
- `results_path(aspect)` / `log_result(aspect, record)` — appends the JSON result line to `results/<aspect>/results.jsonl` (adds a timestamp)
- `hb(msg)` — timestamped heartbeat print, also appended to `results/_progress.log` (this is why SLURM logs are greppable by time)

---

## Layer 2 — per-aspect mains and their model classes

### Aspect 1 — directionality

**`a1_directionality.py`** — defines `GNNModel` (line 42): HeteroEncoder +
HeteroTemporalEncoder + HeteroGraphSAGE + linear head, with the
**directionality switch in `forward()`** (lines 96–120):
- `U`: all edges (undirected — `make_pkey_fkey_graph` already added `rev_` edges)
- `D`: filters out every edge type whose relation name starts with `rev_` (line 101–102)
- `DirGNN`: **two parallel SAGE stacks** (`self.gnn` fwd / `self.gnn_bwd` bwd, built at lines 75–80) merged by `nn.Linear(2*channels, channels)` — the merge happens at lines 111–119
- Design detail (line 67–70 comment): all variants build the **full** edge-type module set and filter only at forward time, so U/D/DirGNN differ in routing, not module count

### Aspect 2 — heterogeneity (the 2×2 matrix)

- **`a2_heterogeneity.py`** — `Backbone` (line 40): same structure as `GNNModel` but the stack is a `build_gnn(kind)` switch, `'sage'` (HeteroSAGE row) or `'hgt'` (HGT-Hetero row)
- **`a2_hgt.py`** — HGT-only rerun reusing `Backbone`, after the first HGT run got stuck at AUC 0.500 (see its docstring — this is the no-external-ReLU fix history)
- **`a2_homo_sage.py`** — `HomoSAGEModel` (line 44): the **homogeneity collapse**. Per-type Frame encoders stay (each table has a different column schema, so a shared encoder is impossible — matches the report's implementation note), then: concatenate all node embeddings into one tensor, **shift edge indices by per-type offsets**, concat into one `(2, E)` edge tensor, run plain `SAGEConv`, slice the target rows back out by offset
- **`a2_hgt_homo.py`** — `HGTHomo` (line 45): same collapse with `HGTConv` on synthetic metadata `(["node"], [("node","edge","node")])` (line 67). The collapse steps are commented 1–5 in `forward()` (lines 74–120)
- **`a2_hgt_tuned.py`** — `HGTTuned` (line 54): the ‡ tuning recipe — **hidden 256, heads 8, epochs 30, LR 1e-3 + CosineAnnealingLR** (docstring lists the exact diffs vs base); tests whether SAGE>HGT is an under-tuning artifact
- **`a2_tuned.py` / `a2_homo_sage_tuned.py` / `a2_hgt_homo_tuned.py`** — the same recipe applied to the other three cells of the 2×2 ("full 2×2 tuning" bullet in the report)

### Aspect 3 — node feature encoders

- **`a3_encoders.py`** — `EncoderAblation` (line 44): the **id/column/llm switch** (lines 53–60) selecting between the three encoder classes from `pipeline.py`, on the HeteroSAGE backbone. Subtlety at lines 62–68: the temporal encoder is kept even for `id`, otherwise id-only has no notion of time at all
- **`a3_hgt.py`** — `EncoderAblationHGT` (line 40): identical but with the HGT backbone — **this is the spec-required primary A3**; the SAGE version is the reference comparison
- **`a3_tuned.py`** — all 12 (dataset × backbone × encoder) configs under the tuning recipe (hidden 256, LR 1e-3 cosine, 20 epochs); contains `train_and_eval_cosine()`
- **`a3_multiseed.py`** — rel-hm × HeteroSAGE × {id, column, llm} at seeds {123, 2024} (seed 42 comes from the base run) — the 3-seed evidence that column-vs-llm is within noise

### Aspect 4 — depth and oversmoothing

- **`a4_depth.py`** — `DepthModel` (line 41): depth-configurable (`n_layers`), **sum aggregation**, optional **residual skip** — previous representation added back after each HeteroConv, before LayerNorm+ReLU (identity residual; see docstring)
- **`a4_mean.py`** — `DepthMeanModel` (line 52): the **spec-required mean aggregation** version (`AGGR = "mean"`, line 37), plus the oversmoothing diagnostic:
  - **`_mad()` (line 119)** — mean pairwise cosine distance over ≤512 sampled target embeddings: normalize, `n @ n.T`, mean of off-diagonal `1 − sim`
  - per-layer collection via the `self._collect_mad` flag inside `forward()` (lines 97–114), triggered once per depth by `collect_mad_on_val()` (line 134)
  - `_capped_neighbors()` (line 47) — fan-out `[128, 32, 5, 5, …]` and batch 128 so deep stacks fit in memory (the batch-size confound noted in the report tables)
- **Fill-in wrappers** (all layer 3, each docstring says exactly which SLURM job failed and why): `a4_fits.py`, `a4_missing.py`, `a4_hm_cont.py` (fan-out fix after NeighborLoader hung on rel-hm's 30M-row transactions table at L≥5), `a4_hm_deep.py`, `a4_stack_*.py`, `a4_mean_*.py`, `a4_skip_*.py`
- **`a4_l16.py` / `a4_l16_h200.py`** — the L=16 attempts that OOM'd (the >126 GB activation tensor story)
- **`a4_tuned.py` / `a4_sum_tuned.py` / `a4_mean_tuned.py`** — tuning checks at the best depths

### t-SNE and multiseed (Aspect 1 extras)

- **`a1_tsne.py`** — trains each A1 variant briefly on rel-stack, dumps 5,000 pre-head target embeddings + labels to `results/a1_tsne/{U,D,DirGNN}.npz`; the notebook's t-SNE cell reads these. Rel-stack chosen over rel-hm because its val split is smaller and the 5%/95% split makes a 2-colour plot informative (82%-positive rel-hm would be one colour)
- **`a1_multiseed.py`** (+ `_tail.py`) — the 3-seed A1 validation (seeds 42/123/2024); `_tail` finishes the two configs the main job lost to a SLURM timeout

### Sanity probe (pre-RelBench generation, kept because it's cited in the report)

- **`rel_hm_reweighted.py`** — the class-reweighting probe: re-runs HomoSAGE and HGT-Hetero on rel-hm with `pos_weight = n_neg/n_pos` in the BCE loss. Result (AUC 0.500 / 0.528 — near random) is the "reweighting probe" bullet in the report's Sanity checks. Built on **`../src/`**: `data.py` (`build_data`), `features.py` (HW1-style column featurization), `loaders.py` (NeighborLoader wiring), `models.py` (its own `HomoSAGE`/`HGT` classes), `train.py` (`fit`, `TrainConfig`)

---

## "Where is X implemented?" lookup

| Question | File | Symbol / line |
|---|---|---|
| Graph construction (DB → heterograph) | `pipeline.py` | `build_graph_data`, L92; cache at L104 |
| Neighbor sampling / temporal split handling | `pipeline.py` | `build_loaders`, L114; fan-out L122 |
| Training loop, loss, optimizer | `pipeline.py` | `train_and_eval`, L344 |
| `pos_weight` | `pipeline.py` | `compute_pos_weight_from_task`, L331; passed to BCE at L356 |
| Metrics (AUC, AUPRC, P, R, F1) | `pipeline.py` | `compute_metrics`, L299 |
| F1\* threshold grid | `pipeline.py` | `find_best_f1`, L312 |
| MCC / Balanced Accuracy | `notebook.py` | `mcc_ba_from_pr()` (~L294) + `explode_val()` (~L329) — not in pipeline |
| id / column / llm encoders | `pipeline.py` | `IdEncoder` L176, `ColumnEncoder` L199, `LLMEncoder` L224 |
| Text embedding + NaN handling | `pipeline.py` | `SentenceEmbedder`, L61 |
| SAGE vs HGT construction | `pipeline.py` | `build_gnn` L245, `run_gnn` L270 |
| Why HGT has no ReLU between layers | `pipeline.py` | `run_gnn` docstring, L273 |
| U / D / DirGNN switch | `a1_directionality.py` | `GNNModel.forward`, L96–120 |
| DirGNN's fwd/bwd stacks + merge | `a1_directionality.py` | L75–80 (build), L111–119 (merge) |
| Empty-edge / isolated-node robustness | `pipeline.py` | `safe_hetero_forward`, L157 |
| Homogeneity collapse (concat + index shift) | `a2_homo_sage.py` / `a2_hgt_homo.py` | `forward()` steps 1–5 |
| HGT tuning recipe (‡ in Table 2) | `a2_hgt_tuned.py` | header docstring + `HGTTuned` L54 |
| Depth sweep + skip connections | `a4_depth.py` | `DepthModel`, L41 |
| **MAD (oversmoothing diagnostic)** | `a4_mean.py` | `_mad` L119, collection in `forward` L97–114 |
| Mean-aggregation switch | `a4_mean.py` | `AGGR = "mean"`, L37 |
| Memory workarounds for deep L | `a4_mean.py` `_capped_neighbors` L47; `a4_fits.py`, `a4_missing.py`, `a4_hm_cont.py` docstrings |
| L=16 OOM attempts | `a4_l16.py`, `a4_l16_h200.py` | |
| Multi-seed runs | `a1_multiseed.py`, `a3_multiseed.py` | module-level `SEEDS` |
| t-SNE embedding dumps | `a1_tsne.py` | writes `results/a1_tsne/*.npz` |
| Class-reweighting sanity probe | `rel_hm_reweighted.py` | uses `../src/` |
| Result logging (JSONL) | `common.py` | `log_result`, L31 |
| Where results land | `results/<aspect>/results.jsonl` | aspect name = first arg of `log_result` in each main |
| How the notebook reads results | `notebook.py` | `load_results(aspect)`; project root autodetected by `results/` + `code/` dirs |

## Anatomy of one run (what happens when a script executes)

```
python code/a3_encoders.py
 └─ common.setup_paths()                  # cwd → project root
 └─ pipeline.build_graph_data(...)        # cached graph (results/graph_cache)
 └─ pipeline.compute_pos_weight_from_task # 0.22 on rel-hm
 └─ for encoder in [id, column, llm]:
      torch.manual_seed(42)               # same seed per variant
      pipeline.build_loaders(...)         # fresh loaders per variant
      model = EncoderAblation(...)        # only the studied axis differs
      pipeline.train_and_eval(...)        # 10 epochs, keep best-AUC epoch
      common.log_result("a3_encoders", {config, val metrics, F1*, params, wall time})
 └─ results/a3_encoders/results.jsonl     # one JSON line per variant
```
