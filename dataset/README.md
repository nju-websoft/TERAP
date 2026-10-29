# Datasets

This directory contains taxonomy snapshots and query splits for Admission and Placement, included directly in the repository. The folder is about 250 MiB. For a smaller download, you can instead use [dataset.tar.xz](../dataset.tar.xz) (about 22 MiB), which contains the same files and can save download time. Extract it from the repository root with `tar -xJf dataset.tar.xz`; no extraction is needed if you already have this directory.

## Benchmarks and splits

Counts below are **JSONL rows (queries)**, not numbers of unique concepts.

| Dataset | Task | Train | Valid | Test |
| --- | --- | ---: | ---: | ---: |
| TaxoAP-MeSH | Admission | 27,185 | 7,146 | 3,710 |
| TaxoAP-MeSH | Placement | 32,768 | 171 | 116 |
| TaxoAP-IEEE | Admission | 6,731 | 4,695 | 2,701 |
| TaxoAP-IEEE | Placement | 14,177 | 65 | 210 |
| QEN-MeSH | Placement | 3,848 | 550 | 1,100 |
| IEEE-static | Placement | 1,752 | 250 | 500 |

- **TaxoAP-MeSH** is a temporal benchmark. Validation and test queries use years 2025 and 2026, respectively.
- **TaxoAP-IEEE** is a temporal benchmark. Validation and test queries use years 2024 and 2025. Its Admission test split contains 217 new-concept examples and 2,484 negatives.
- **QEN-MeSH** is a static MeSH Placement benchmark.
- **IEEE-static** is a static Placement split of the 2025 IEEE taxonomy, also referred to as TaxoAP-IEEE-static.

The TaxoAP-MeSH Admission training total includes 11,480 weak negatives. Query-Only uses this expanded pool; the memory classifier excludes weak negatives and uses 15,705 examples (2,751 new and 12,954 negative). The split file is shared, with filtering applied by the recipe.

## Directory layout

```text
dataset/
├── README.md
├── TaxoAP-MeSH/
│   ├── taxonomy/             # 2015.json through 2025.json
│   ├── admission/            # train.jsonl, valid.jsonl, test.jsonl
│   └── placement/            # train.jsonl, valid.jsonl, test.jsonl
├── TaxoAP-IEEE/
│   ├── taxonomy/             # 2019.json through 2024.json
│   ├── admission/            # train.jsonl, valid.jsonl, test.jsonl
│   └── placement/            # train.jsonl, valid.jsonl, test.jsonl
├── QEN-MeSH/
│   ├── taxonomy/static.json
│   └── placement/            # train.jsonl, valid.jsonl, test.jsonl
└── IEEE-static/
    ├── taxonomy/static.json
    └── placement/            # train.jsonl, valid.jsonl, test.jsonl
```

Each JSONL line is one sample. A taxonomy JSON contains a `records` mapping keyed by concept UID, with node names, parent UIDs, ancestor paths, and available descriptive fields.

For a temporal query with `year = Y`, use `taxonomy/<Y-1>.json`. For static tasks, use `taxonomy/static.json`; `year = 0` is the static graph key, not a calendar year. The loaders in `src/data.py` implement this convention.

## Sample fields

Fields vary by task; the following are the main inputs and labels.

| Field | Meaning |
| --- | --- |
| `sample_id` or `id` | Sample identifier; use `src.data.sample_id` for a unified accessor |
| `query` | Model input text, where supplied |
| `year` | Selects the appropriate taxonomy snapshot |
| `label` | Admission label: 1 for new, 0 for negative |
| `weak` | Admission weak-example flag, interpreted by the recipe |
| `target` | Placement target concept UID |
| `gold` | List of correct parent UIDs for Placement |
| `masked` | Whether the target must be excluded from candidate retrieval |
| `paragraph`, `start`, `end` | IEEE Admission context and mention character offsets |

Do not assume every row includes a `split` field. `load_rows` assigns it from the source file and checks any existing value for consistency. Preserve this field when building training caches.

Placement can have multiple correct parents: preserve the full `gold` list. For masked samples, exclude `target` from retrieval candidates and negative candidates. Admission and Placement have different eligible samples and may use different query formatting; join stages by sample ID rather than row position, and keep each task's own text.

## Query text and paragraph context

**Released `query` values are text strings, not token IDs or vectors.** Some strings include serialized `[CLS]` and `[SEP]` markers. `src.data.query_tokens` handles these markers during tokenization.

- **IEEE Admission:** context is in `paragraph`. The mention is `paragraph[start:end]`; the loader uses the mention and surrounding text to construct its token window.
- **MeSH Admission and Placement:** `query` already contains the prepared concept/context text. Some context has been truncated during preparation.
- **IEEE Placement and IEEE-static:** `query` contains prepared context. `article_id` and `paragraph_id` identify its source; they are not a separate full-paragraph corpus.
- **QEN-MeSH:** `query` contains concept and descriptive text, rather than a separate source-paragraph field.

Some masked MeSH training rows include `evidence` provenance, but this is not a copy of the full source paragraph. The provided text supports the released input pipeline; reconstructing different context windows may require source text beyond this archive.

The processing stages are:

```text
query text (or paragraph + mention offsets)
    → tokenizer → token IDs → encoder → feature vectors
```

Generated token and feature caches belong to the working directory. They are not included in this dataset archive.

## Loading the data

From the repository root:

```python
from src.data import load_rows, load_taxonomy, sample_id

root = "dataset/TaxoAP-MeSH"
rows = load_rows(root, "placement", "train")
row = rows[0]
records = load_taxonomy(root, year=row["year"])

print(sample_id(row), row["query"])
print(row["gold"])
```

Use `Corpus` with a compatible tokenizer for tokenization and graph indexing. The root [README](../README.md) provides the complete download, configuration, preparation, training and evaluation commands. `python -m src.run --dataset <name>` builds the required caches from these files before invoking the existing recipes. Keep validation and test examples out of gradient training, and rebuild derived caches when changing retrieval or memory settings.

The release includes raw split files and taxonomy graphs. It does not include manifests, separate metadata files, tokenized copies, pretrained weights, or trained checkpoints.
