# TERAP

Anonymous research code for taxonomy admission and placement.

This repository provides the core models, task-specific training recipes, and inference and evaluation utilities. **Admission** decides whether a proposed concept should enter a taxonomy. **Placement** ranks its possible parent concepts. The implementation includes bi-encoder (BE) retrieval, cross-encoder (CE) reranking, query-only admission, and taxonomy-memory models.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── dataset.tar.xz
├── dataset/                  # Dataset files included in the repository
│   ├── README.md
│   ├── TaxoAP-MeSH/
│   ├── TaxoAP-IEEE/
│   ├── QEN-MeSH/
│   └── IEEE-static/
└── src/
    ├── admission.py          # Admission heads and models
    ├── placement.py          # Placement models
    ├── placement_ieee.py     # IEEE Placement model components
    ├── encoder.py
    ├── retrieval.py          # BE and CE components
    ├── memory.py             # Taxonomy-path memory construction
    ├── data.py               # Dataset loading and tokenization
    ├── pipeline.py           # Input preparation and prediction
    ├── evaluation.py         # Admission and Placement metrics
    ├── configuration.py      # Per-task configuration loading
    ├── settings.py           # Model settings and candidate text styles
    ├── train.py              # Training launcher
    └── recipes/              # Task implementations and their configs/
```

## Installation and data

Run commands from the repository root in a Python environment compatible with the dependencies:

```bash
python -m pip install -r requirements.txt
```

The dataset files are included directly in `dataset/` and can be used without extraction. Since this folder is relatively large (about 250 MiB), you can alternatively download [dataset.tar.xz](dataset.tar.xz) (about 22 MiB) and extract it locally. This optional archive contains the same files and can save download time when you only need the data. From the repository root, run:

```bash
tar -xJf dataset.tar.xz
```

The archive uses XZ compression with a 16 MiB dictionary. The dictionary is embedded in the compression format; no external dictionary file is needed. On Windows, an XZ-capable archive utility can extract the `.xz` layer and then the `.tar` archive.

See [dataset/README.md](dataset/README.md) for split sizes, fields, and text inputs. The archive contains source data only. Pretrained weights, trained checkpoints, and generated token/feature caches are not included.

Place compatible BERT weights and tokenizer files in `models/bert/`, or override the relevant recipe paths. The launcher defaults to offline model loading; download the required model files before running it. Training recipes use CUDA.

## Configuration

Each task has its own JSON file:

```text
src/recipes/<implementation>/configs/<recipe>.json
```

**Configuration values are intentionally `null`. Fill the selected task's configuration before running it.** These files are templates, not ready-to-run experiment presets. The launcher rejects unfilled training fields. Architecture and objective definitions also reside in the model and recipe code.

The `training` section holds the settings relevant to that task, such as learning rates, optimizer, seed, batch size, steps or epochs, and validation frequency. `eval_every` counts optimizer steps; `eval_every_epochs`, where present, counts epochs. IEEE Admission's `projection_start` is a zero-based optimizer step, independent of the BERT unfreezing step `switch`.

For memory-based tasks, `inputs` contains separate `training` and `inference` sections:

| Field | Meaning |
| --- | --- |
| `be_k` | Number of BE candidates passed to CE reranking |
| `ce_k` | Number of reranked candidates retained downstream |
| `max_paths` | Maximum number of active memory paths |
| `slots` | Padded memory width |

Require `ce_k <= be_k` and `max_paths <= slots`. Fill both phases before using both training and inference. Changing training retrieval or memory sizes requires regenerating the affected caches; the launcher does not rebuild them.

## Training recipes

| Component | Recipe names | Implementation directory |
| --- | --- | --- |
| MeSH Admission | `mesh_query`, `mesh_memory` | `mesh_query`, `mesh_memory` |
| IEEE Admission | `ieee_query`, `ieee_memory` | `ieee_query`, `ieee_memory` |
| QEN-MeSH path encoder | `qmesh_encoder` | `qmesh_encoder` |
| Placement | `placement_mesh`, `placement_ieee`, `placement_qmesh`, `placement_static` | Same as recipe name |
| MeSH Placement retrieval | `mesh_be`, `mesh_ce` | `retrieval_mesh` |
| IEEE / QEN-MeSH Placement retrieval | `ieee_be`, `ieee_ce`, `qmesh_be`, `qmesh_ce` | `retrieval_temporal` |
| Static IEEE retrieval | `static_be`, `static_ce` | `retrieval_static` |
| IEEE Admission retrieval | `ieee_admission_be`, `ieee_admission_ce` | `retrieval_ieee` |
| IEEE Admission mining / cache preparation | `ieee_admission_mining`, `ieee_admission_cache` | `retrieval_ieee` |

After filling the configuration and preparing the required inputs:

```bash
# Inspect the resolved command and training settings without starting training.
python -m src.train placement_ieee --show

# Run with the recipe's configuration.
python -m src.train placement_ieee

# Select an alternative configuration file.
python -m src.train placement_ieee --config path/to/config.json
```

`--show` also validates required settings, so it requires a filled configuration.

The usual dependency order is retrieval training, candidate/feature preparation, then downstream model training. Admission also requires its query-only and memory branches; Placement uses CE features and encoded taxonomy paths. Recipes consume prepared inputs rather than providing a single command that converts every raw dataset into all training artifacts.

## Working directories and caches

Generated inputs and outputs normally live under `work/<implementation>/`. Temporal retrieval adds an `ieee` or `qmesh` subdirectory. Paths can be controlled with:

- `RELEASE_ROOT`: repository/data root.
- `RELEASE_WORK`: base directory for working artifacts.
- `RELEASE_PATHS`: JSON mapping from implementation name to path-variable overrides used in its `common.py`.

For example, to place generated artifacts outside the repository:

```bash
export RELEASE_WORK=/path/to/work
```

Admission recipes read prepared arrays from their `data/` directory. Retrieval recipes use task-specific row, token, and graph artifacts; their exact filenames are defined in the corresponding implementation. Raw JSONL files are not interchangeable with these caches.

Placement uses the following layout:

```text
work/<placement implementation>/cache/s<seed>/
├── rows.json
├── inputs.npz
├── features.npy
├── path_features.npy
└── CE_head.pt
```

`inputs.npz` contains `memory`, `roles`, `gold`, and `hard8`. Rows must contain `target`, explicit `split`, and `gold_indices` (all correct parent indices in the graph candidate bank, including parents outside the shortlist). Convert the raw UID-based `gold` labels using the same candidate index used to build the features. Rows must explicitly identify their split and follow train/valid/test order. Test rows may be included in the cache for later evaluation; gradient training uses only train rows, and checkpoint selection uses only validation rows. Feature arrays must follow the same row order, and memory indices must reference the supplied path-feature bank. Optional `valid_mrr_tail` enables full validation MRR reporting; without it, the cache-based metric is truncated.

## Inference and evaluation

The public API is organized around:

- `src.data.Corpus`: load splits, select taxonomy snapshots, and tokenize text.
- `src.pipeline.RetrievalEngine`: retrieve and rerank parent candidates.
- `prepare_admission` / `prepare_placement`: construct task inputs using the selected configuration.
- `predict_admission` / `predict_placement`: produce scores or parent rankings.
- `src.evaluation.evaluate_admission` / `evaluate_placement`: evaluate prepared examples.
- `src.pipeline.end_to_end`: join Admission and Placement predictions by sample ID.

Use the candidate text style in `src.settings.CANDIDATE_STYLES` for the task. In particular, IEEE Admission uses `ieee_paths`, while IEEE Placement uses `name_definition`. Load compatible checkpoints and construct the model with the same architecture used during training. `model_settings` reads the task configuration; IEEE Admission head checkpoints additionally store `model_config`.

Training runs for the configured steps or epochs, without validation-based early stopping. Validation is used to select the checkpoint. Placement text encoders remain frozen. The release contains the main method, including both Admission cascade branches; ablation-specific training and model variants are not included. Retrieval recipes may export test candidates once after training from the selected validation checkpoint for downstream evaluation. Final test evaluation is separate from the training validation loop.

Admission evaluation selects cascade thresholds on validation and transfers them to test, reporting new-concept recall, negative rejection, AUROC, and partial AUROC. Placement reports H@K, R@K, and MRR, averaging queries within each target concept before averaging targets. Full-ranking MRR requires the ranking tail; it should not be confused with MRR@32.
