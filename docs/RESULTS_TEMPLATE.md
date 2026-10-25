# Results report template — FGVC-Aircraft

> **UNMEASURED TEMPLATE.** Replace every “TBD” only with an actual benchmark result linked to its run artifacts. Do not copy example or estimated performance values into the report.

## 1. Experimental context

| Field | Recorded value |
| --- | --- |
| Report date / git commit | TBD |
| Dataset edition (Torchvision 100-variant or other) | TBD |
| Dataset manifest SHA256 | TBD |
| Taxonomy SHA256 | TBD |
| Train / validation / test counts | TBD |
| Image preprocessing / resolution | TBD |
| Training seeds | TBD |
| Framework, CUDA and GPU driver | TBD |
| Inference hardware and GPU memory | TBD |
| Power mode / model residency | TBD |
| Dtype and batch size | TBD |
| Preprocessing, latency and FPS benchmark boundaries | TBD |
| Experiment configuration SHA | TBD |
| Result artifacts / locations | TBD |

**Protocol deviations:** TBD. Note missing branches, failures, OOM, class-index changes, data changes, and any post-hoc experiment additions.

## 2. Main comparison

The primary comparison is **IND-2 versus FT-FROZEN-2** on identical test images. Variant macro recall and p95 full-pipeline latency are the pre-registered endpoints.

| Metric | FLAT | IND-2 | FT-FROZEN-2 | FROZEN-2 | IND-3 | FT-FROZEN-3 | FROZEN-3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Variant macro recall (%) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Variant top-1 (%) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Variant macro-F1 (%) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Manufacturer accuracy from final variant (%) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Family accuracy from final variant (%) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Oracle variant macro recall (%) | N/A | TBD | TBD | TBD | TBD | TBD | TBD |
| First-stage routing accuracy (%) | N/A | TBD | TBD | TBD | TBD | TBD | TBD |
| Full pipeline p50 (ms) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| **Full pipeline p95 (ms)** | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Model-path p95 (ms) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Continuous throughput (FPS) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Peak GPU allocated memory (MiB) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Total deployment artifact size (MB) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Total model parameters (M) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Mean executed MACs / image (G) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Cumulative training time / GPU-hours | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Use `mean ± standard deviation` across seeds for accuracy; explicitly state how aggregate latency was combined across repeated passes and seeds. Distinguish MiB from decimal MB and MACs from FLOPs. For FLAT, oracle routing is not applicable.

## 3. Paired differences, uncertainty and engineering objective

| Pre-specified pair | Δ variant macro recall (shared − independent, pp) | Paired 95% bootstrap CI (pp) | Δ p95 latency (shared − independent, ms) | p95 latency reduction (%) |
| --- | --- | --- | --- | --- |
| FT-FROZEN-2 vs IND-2 | TBD | TBD | TBD | TBD |
| FT-FROZEN-3 vs IND-3 | TBD | TBD | TBD | TBD |
| FROZEN-2 vs IND-2 | TBD | TBD | TBD | TBD |

**Engineering decision check:** Compare the 2-stage shared model against the exploratory limits of **≤2 pp macro-recall degradation** and **≥30% p95 latency reduction**. Report *whether observed values and uncertainty support the claim*. Do not claim noninferiority simply because the observed difference is within 2 pp.

**Paired bootstrap detail:** Sampling scheme, number of replicates, per-seed vs across-seed aggregation, CI type, and selected model seed: TBD.

## 4. Routing error attribution

| Architecture | Manufacturer router correct (%) | Conditional family router correct (%) | Hard-route variant macro recall (%) | Oracle-route variant macro recall (%) | Oracle − hard penalty (pp) |
| --- | --- | --- | --- | --- | --- |
| IND-2 | TBD | N/A | TBD | TBD | TBD |
| FT-FROZEN-2 | TBD | N/A | TBD | TBD | TBD |
| IND-3 | TBD | TBD | TBD | TBD | TBD |
| FT-FROZEN-3 | TBD | TBD | TBD | TBD | TBD |

Describe **conditional denominators** and where the first wrong decision occurs. Detail the count of singleton-family shortcuts and their frequency in evaluated samples.

## 5. Latency by path

| Architecture | Path depth / branch type | Test images | p50 (ms) | p95 (ms) | Peak memory (MiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| IND-2 | TBD | TBD | TBD | TBD | TBD |
| FT-FROZEN-2 | TBD | TBD | TBD | TBD | TBD |
| IND-3 | TBD | TBD | TBD | TBD | TBD |
| FT-FROZEN-3 | TBD | TBD | TBD | TBD | TBD |

Attach latency histograms, breakdown of preprocessing/backbone/router/specialist/transfer, and accuracy-versus-p95 Pareto figure with units.

**Cold-start vs steady-state:** TBD. **Model residency / any swap and copy costs:** TBD.

## 6. Specialist update / introduction

| Metric | Independent cascade | Shared frozen backbone |
| --- | --- | --- |
| Original known class count | TBD | TBD |
| Introduced variant and manufacturer | TBD | TBD |
| New-class recall after update (%) | TBD | TBD |
| Updated branch's existing-class macro recall before/after (%) | TBD | TBD |
| Unaffected families' accuracy before/after (%) | TBD | TBD |
| Specialist update wall time (s) | TBD | TBD |
| Parameters retrained | TBD | TBD |
| Extra data used | TBD | TBD |
| Replacement artifact size (MB) | TBD | TBD |
| Backbone and unrelated head checksum unchanged? | TBD | TBD |

Confirm explicitly that the introduced variant was **excluded from the initial training of any backbone**, not merely omitted from specialist heads. Document changes in routing policy or taxonomy.

## 7. Plots to publish

- Variant confusion matrices: IND-2, FT-FROZEN-2, FLAT.
- Within-family and cross-family error breakdown.
- Accuracy vs p95 latency with Pareto frontier and uncertainty where available.
- Per-sample full-pipeline latency distribution, grouped by execution depth.
- Per-class recalls and specialist support counts.
- Model storage and GPU memory comparisons.
- Incremental specialist training-cost and accuracy-before/after plots.

## 8. Discussion and conclusion

**RQ1 — Accuracy:** TBD.  
**RQ2 — Latency:** TBD.  
**RQ3 — Routing error:** TBD.  
**RQ4 — Modularity:** TBD.  
**RQ5 — Generic vs task-adapted frozen features:** TBD.

Distinguish findings supported by measured data from architectural expectations. State whether the result generalizes beyond the particular FGVC-Aircraft class hierarchy and input resolution; avoid claiming generality from one dataset.

## 9. Reproduction commands and audit links

Commands used (not planned commands): TBD.

| Architecture/seed | Git SHA | Config hash | Backbone checkpoint ID | Head bundle ID | Test prediction CSV | Timing CSV | Metrics JSON |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**Known limitations, incomplete runs, unresolved failures:** TBD.