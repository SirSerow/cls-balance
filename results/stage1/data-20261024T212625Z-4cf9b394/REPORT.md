# Stage-one data validation — data-20261024T212625Z-4cf9b394

This report follows the context and audit portions of `docs/RESULTS_TEMPLATE.md`.
Stage one is complete. No classification, training, inference-latency or throughput measurements have been made; the template's model comparison sections do not yet apply.

## 1. Experimental context

| Field | Measured/recorded value |
| --- | --- |
| Audit run | `data-20261024T212625Z-4cf9b394` |
| Implementation Git SHA | `453a4fe8b9ec00e1b953e5accf6549bcbd1786fb` (clean checkout during final audit) |
| Dataset | Complete official Torchvision 100-variant edition |
| Train / validation / test | 3334 / 3333 / 3333, mutually disjoint image IDs |
| Variants / families / manufacturers | 100 / 70 / 30 |
| Images decoded and checksummed | 10000; exact image inventory matched annotations |
| Per-variant total support | 100 images, with all variants represented in each split |
| Manifest-set SHA256 | `40516fd3cf71f4431d692a4b6f5b183a8bb6cae026acef08e2ab2cdfdf92b671` |
| Taxonomy SHA256 | `a79cf6f7fe8a7a028c1f2b51619b307750a241c26d863c0fc983423a6dd48177` |
| Preprocessing SHA256 | `173fd93e7491bb9048ffe5257c364450fa69b3f0e1ac4b1a4cf965eddea78adc` |
| Image-checksum-list SHA256 | `6b123e6c081b19a265a2cff95cc31c433294b4e8166a9602ba323e68b1876b8e` |
| Execution environment | Windows x64; Python 3.12.14; Torch 2.9.1+cpu; Torchvision 0.24.1+cpu |
| Execution device / residency | CPU data preparation; no model resident |
| Seed | 42 |
| Configuration SHA256 | `32aa5dc453cd00678c5c633137ffa85ca2ea4f0f4a1c809a3562caf562691250` |

The complete resolved configuration, preprocessing profile, environment export, source annotation hashes and aggregate support counts are saved beside this report. Split manifests and per-image checksums remain local under `runs/data/data-20261024T212625Z-4cf9b394/`. No copyrighted images or per-image label files are published.

The taxonomy is derived from training annotations joined by image ID. Validation and test labels only check completeness and hierarchy consistency. Canonical names are sorted from train; IDs explicitly map Torchvision's targets into the global taxonomy.

## 2. Data and preprocessing checks

All 10000 images decode. Widths range from 775 to 1600 pixels; heights range from 413 to 1205 pixels. Every test image has exactly one valid variant → family → manufacturer path. A second full audit reproduced the manifest, taxonomy, preprocessing and per-image checksum-list hashes from the first audit.

Primary preprocessing removes the bottom 20-pixel banner, preserves aspect ratio using bicubic resize, centers the remaining image in a black 224×224 square, converts to float32 and normalizes with EfficientNet-B0 ImageNet V1 mean/std. Six training-only visual pairs were inspected; no added clipping was observed. One original is itself a tail close-up. See `visual_review.json`; image previews remain local.

| Branch | Single child | Multiple children | Empty | Test-image true-route bypass potential |
| --- | ---: | ---: | ---: | ---: |
| family_to_variant | 54 | 16 | 0 | 1798 / 3333 |
| manufacturer_to_family | 14 | 16 | 0 | 502 / 3333 |
| manufacturer_to_variant | 13 | 17 | 0 | 435 / 3333 |

These are taxonomy counts and **true-route bypass potential**, not measured predicted-route execution rates. Actual bypass frequency will be measured after training.

## 3. Validation and provenance discrepancy

**56 tests passed**, zero failures/errors/skips, in 8.455 seconds on CPU. The suite includes reordered annotation joins, global IDs, duplicate/missing labels, invalid hierarchy, official split/count guards, missing/extra/corrupt images, letterbox boundary preservation, normalization, two-worker augmentation reproducibility, immutable runs and artifact round trips. Dependency validation: `No broken requirements found.`. The CPU CI workflow is committed; its remote outcome has not been observed for this report.

The official 2,753,340,328-byte archive has MD5 `d4acdd33327262359767eeaa97a4f732`, matching [MILA's independently archived checksum](https://github.com/datasets-mila/datasets--fgvcaircraft/blob/master/md5sums). The [authors' HTML checksum page](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.html) lists `32eca553f897e747706144f3bb444df0`. Cause unknown. Python and PowerShell independently produced the observed MD5; archive extraction and all full-dataset checks succeeded. The CLI enforces the independently recorded archive checksum and retains the authors' discrepancy in `summary.json`.

An initial strict authors-checksum audit (`data-20261024T212341Z-b424c9c3`) failed and wrote no validated summary. It remains local and is not represented as a successful run. This provenance discrepancy is the sole recorded protocol deviation. No dataset records were modified, excluded or substituted.

## 9. Reproduction commands and audit links

Start at implementation commit `453a4fe8b9ec00e1b953e5accf6549bcbd1786fb` and follow the README's Python 3.12 environment installation. Commands executed for validation:

```powershell
.venv/Scripts/python.exe -m pytest -q --basetemp=.tmp/pytest-5 --junitxml=.tmp/unit-tests.xml
.venv/Scripts/python.exe -m pip check
.venv/Scripts/python.exe -m cls_balance.prepare --config configs/data.yaml --no-download
```

The original acquisition used the same prepare command without `--no-download`. New invocations create unique run directories; compare the manifest, taxonomy, preprocessing and per-image checksum-list hashes rather than the run ID or machine-specific environment/config paths.

- [Aggregate dataset audit](summary.json)
- [Test and repeat-audit evidence](validation.json)
- [Visual inspection record](visual_review.json)
- [Environment](environment.json)
- [Resolved configuration](config.resolved.json)
- [Preprocessing](preprocessing.json)
- [Source annotation hashes](source_annotations.json)

Known limitations: model training and all accuracy/latency/resource endpoints remain unmeasured. Checkpoint, routing and frozen-weight tests belong to their later implementation phases. Data and image previews stay in ignored local storage and are not pushed to Git.
