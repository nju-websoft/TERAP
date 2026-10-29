# TERAP

Research code for taxonomy Admission and Placement. Admission combines a query-only classifier with a taxonomy-memory classifier; Placement retrieves and reranks parent candidates using path memory. The release includes the core models, original task training objectives, dataset preparation, and final evaluation.

## Setup

Use Python 3.11 and a CUDA-enabled PyTorch installation on a Linux machine with one visible NVIDIA GPU. Training uses BF16; use a BF16-capable GPU. Preparation and evaluation also accept `--device cpu`, but full training requires CUDA. Model weights, features and checkpoints require additional disk space beyond the dataset; memory requirements depend on the configured candidate counts and batch sizes.

From the repository root:

```bash
python -m pip install -r requirements.txt
export CUDA_VISIBLE_DEVICES=0
python -c "import torch; print(torch.__version__); assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported()"
```

The four datasets are included in `dataset/`. Alternatively, download [dataset.tar.xz](dataset.tar.xz) (about 22 MiB) instead of the approximately 250 MiB directory, then run `tar -xJf dataset.tar.xz`. This is an optional faster download of the same data. See [dataset/README.md](dataset/README.md) for fields and splits.

Download the 768-dimensional [BERT-base uncased](https://huggingface.co/google-bert/bert-base-uncased) model and tokenizer once:

```bash
HF_ENDPOINT=https://hf-mirror.com python -m src.workflow download
```

Omit `HF_ENDPOINT` to download directly from Hugging Face. Files are stored in `models/bert/`; subsequent preparation, training and evaluation load them locally. On an offline server, run the download on another machine and copy this directory together with the repository.

## Run the full workflow

Generate editable starting configurations:

```bash
python -m src.workflow init-configs
```

This creates one JSON per task under `work/configs/`. These are **suggested starting parameters, not the settings used to produce the paper's reported results**. They use AdamW, seed 42, retrieval learning rate `1e-5`, downstream head learning rate `1e-4`, and validation-based checkpoint selection. The original unfilled templates remain beside each recipe. Change the generated task files before starting if needed; `init-configs` refuses to overwrite existing configurations.

Run one dataset, or all four:

```bash
# Optional: inspect the exact commands and their numbering.
python -m src.run --dataset mesh --plan

# MeSH Admission + Placement, including all preparation and evaluation.
python -m src.run --dataset mesh

# Alternatively, run all tasks in a fresh working directory.
# python -m src.run --dataset all
```

Accepted dataset names are `mesh`, `ieee`, `qmesh`, and `static`. `mesh` and `ieee` run both Admission and Placement; the static datasets run Placement. QEN-MeSH first prepares and trains its dedicated path encoder using MeSH Admission with its held-out concepts and aliases excluded. A single invocation with `--dataset all` shares prerequisite stages instead of repeating them.

The runner invokes the existing training recipes in this order:

1. Tokenize the raw data and construct graph indexes.
2. Train BE, mine candidates, and train CE. IEEE Admission uses its own historical-graph retriever.
3. Prepare Admission tokens, frozen initial features and natural path memory; train the two cascade branches where applicable.
4. Prepare Placement CE features and path features; train the Placement model. MeSH and QEN-MeSH use their fine-tuned path encoders; IEEE variants use base BERT path features.
5. Load validation-selected checkpoints, evaluate held-out data, and write metrics and predictions.

Final outputs are in `work/results/`: `<dataset>_admission.json` and/or `<dataset>_placement.json`, with sibling `.predictions.jsonl` files containing sample IDs and scores or rankings. Admission thresholds are fitted on validation and transferred unchanged to test. Placement metrics average queries within each target before averaging targets. The two tasks have different eligible examples; their predictions can be joined by ID for shared samples, but the supplied splits are not a one-to-one cascade benchmark.

The runner stops on an error. Fix the reported cause and use the numbered plan to continue, for example `python -m src.run --dataset mesh --start-at 5`. Keep the same configs and completed artifacts when continuing. For a separate run, set `RELEASE_WORK` to a new directory **before** generating its configurations:

```bash
export RELEASE_WORK="$PWD/work_second_run"
python -m src.workflow init-configs
python -m src.run --dataset static
```

## Configuration and individual stages

Each training task has a separate config. `training` controls learning rates, optimizer, batch sizes, steps/epochs and validation intervals. `inputs` separates training and inference candidate/memory sizes. The starting configs use BE top-256, CE top-32, and 32 memory slots in both phases. Require `ce_k <= be_k` and `max_paths <= slots`; rebuild caches when changing input sizes, tokenizer or upstream checkpoints. The retrieval recipes retain their own validation ranking cutoffs, independent of downstream memory sizes.

Admission `switch` and IEEE `projection_start` are zero-based update indices; the starting configs use zero to produce fine-tuned encoders for downstream path encoding. `eval_every` counts optimizer steps and `eval_every_epochs` counts epochs. MeSH BE and CE must share seed and epoch count because the CE recipe locates its BE checkpoint by those values. Use validation to tune parameters.

You can execute any command printed by `--plan` separately. For example:

```bash
python -m src.workflow retrieval-data --dataset ieee
python -m src.train ieee_be --config work/configs/ieee_be.json
python -m src.train ieee_ce --config work/configs/ieee_ce.json
```

`python -m src.workflow --help` lists preparation and evaluation stages. `--be`, `--ce`, `--path-checkpoint` and the final checkpoint arguments select trained weights explicitly. `--batch-size` controls preparation/evaluation batching, independently of recipe training batch sizes. Training may still need a smaller `micro` or `chunk` when GPU memory is limited. Placement currently loads its prepared feature banks into GPU memory.

## Repository layout

```text
src/
├── admission.py, placement.py, placement_ieee.py  # Models
├── encoder.py, retrieval.py, memory.py           # Shared components
├── data.py, pipeline.py, evaluation.py           # Inputs and prediction APIs
├── workflow.py                                  # Raw data preparation and final evaluation
├── run.py                                       # Ordered workflow commands
├── train.py, configuration.py, settings.py      # Training launcher and settings
└── recipes/                                     # Task implementations and config templates
work/                                            # Generated; not distributed
├── configs/
├── <implementation>/                            # Tokens, caches and selected checkpoints
└── results/
```

Training preserves each recipe's loss and sampling rules, runs for configured steps/epochs, and selects checkpoints using validation. Test is not evaluated in training validation loops. Retrieval may export final test candidates after selecting a checkpoint. No experiment dashboard, ablation variants, pretrained weights or trained checkpoints are included.

## Small integration test

The integration test uses synthetic data and a randomly initialized one-layer BERT with the required 768-dimensional interface. It tests execution and checkpoint compatibility, not model quality:

```bash
python tests/make_fixture.py /tmp/terap-smoke
cp -r tests /tmp/terap-smoke/
CUDA_VISIBLE_DEVICES=0 python /tmp/terap-smoke/tests/run_fixture.py
```

Use a new destination each time. The test runs the actual preparation and training commands for all six task/dataset combinations and checks that final validation/test results exist. Full experiments use the released datasets and downloaded BERT weights, with the commands above.
