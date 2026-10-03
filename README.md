# TCM-DPTIR

Source code for the graph–sequence drug–target interaction (DTI) ranking component used in the TCM-DPTIR workflow. This repository contains model and training/inference code only. It does **not** contain trained checkpoints, biomedical datasets, compound libraries, target sequences, or generated prediction results.

## Model overview

The GCADTI-based model encodes compounds as molecular graphs with a graph convolutional network and protein sequences with an embedding plus 1D convolutions. Bidirectional mutual attention conditions each representation on the other before a binary interaction score is produced. The score is a computational prioritization signal; it is not experimental confirmation of binding or activity.

The implementation retains the model and data-loader modules from the current DTI workflow and exposes small, data-agnostic command-line entry points. Upstream GCADTI/DTLCDR attribution is recorded in [`NOTICE.md`](NOTICE.md).

## Repository contents

```text
src/tcm_dptir/       Model, dataset, training and prediction modules
scripts/train.py     Train one or more seeds and select by validation ROC-AUC
scripts/predict.py   Score compound–target pairs with a supplied checkpoint
```

No local CUDA/DGL binaries or vendored third-party packages are included.

## Environment

Python 3.8 or newer is required. Install the PyTorch and DGL builds that match your platform and CUDA runtime, then install the remaining dependencies:

```bash
python -m venv .venv
# Activate the environment using the command for your operating system.
python -m pip install --upgrade pip
python -m pip install torch dgl
python -m pip install -r requirements.txt
python -m pip install -e .
```

For GPU use, follow the official PyTorch and DGL installation instructions to select compatible wheels. The package does not bundle CUDA libraries.

## Input tables

Training and validation CSVs require these columns:

| Column | Description |
| --- | --- |
| `smiles` | Valid compound SMILES string |
| `sequence` | Protein amino-acid sequence |
| `label` | Binary interaction label (`0` or `1`) |

An optional `sampling_weight` column enables weighted sampling during training. If a test CSV is supplied, it is evaluated only after the best epoch is selected on the validation set.

For prediction, the input CSV requires `smiles` and `sequence`; an optional `label` column is ignored for scoring. All other columns are retained in the output, with `gcadti_score` appended.

## Train

```bash
python scripts/train.py \
  --train data/train.csv \
  --validation data/validation.csv \
  --test data/test.csv \
  --output-dir outputs/run1 \
  --device cuda:0 \
  --epochs 50 \
  --seeds 7 21 42 77 2026 \
  --max-drug-nodes 64
```

The test table is optional. The script selects the run and epoch by validation ROC-AUC, then reports optional test metrics without using them for selection. Training writes state dictionaries and run metrics below the chosen output directory. Keep input tables and generated outputs outside version control.

## Predict

```bash
python scripts/predict.py \
  --pairs data/candidate_pairs.csv \
  --checkpoint outputs/run1/gcadti_best.pt \
  --output outputs/candidate_scores.csv \
  --device cuda:0 \
  --max-drug-nodes 64
```

Use the same `max-drug-nodes` and `max-protein-len` values that were used to prepare/train the checkpoint. A checkpoint is not distributed with this source-only repository.

## Reproducibility and interpretation

- Model selection uses validation ROC-AUC; the optional test set is not used to select the epoch.
- Use documented, independently prepared train/validation/test splits and report how unknown-pair negatives were sampled.
- Scores prioritize pairs for follow-up and should be interpreted with external evidence and experimental validation.
- Data acquisition, target mapping, structure/ADME screening, molecular docking, and molecular dynamics are outside this minimal model-code release.

## License status

This repository is public for inspection. No open-source license is included, so no reuse or redistribution rights are granted by this repository at this time. See [`NOTICE.md`](NOTICE.md) for provenance. Add a license only after confirming the rights and intended terms for the adapted upstream component.

