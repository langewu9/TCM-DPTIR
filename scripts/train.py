from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from tcm_dptir.training import train_gcadti


REQUIRED_COLUMNS = {"smiles", "sequence", "label"}


def read_training_table(path: Path, split_name: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"{split_name} table is missing required columns: {sorted(missing)}")
    if not table["label"].isin([0, 1]).all():
        raise ValueError(f"{split_name} labels must be binary values 0 or 1")
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GCADTI and select by validation ROC-AUC")
    parser.add_argument("--train", required=True, type=Path, help="Training CSV")
    parser.add_argument("--validation", required=True, type=Path, help="Validation CSV")
    parser.add_argument("--test", type=Path, help="Optional held-out test CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/run"))
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-drug-nodes", type=int, default=64)
    parser.add_argument("--max-protein-len", type=int, default=1200)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--disable-sampling-weights", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = read_training_table(args.train, "train")
    validation = read_training_table(args.validation, "validation")
    test = read_training_table(args.test, "test") if args.test else None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for seed in args.seeds:
        run_dir = args.output_dir / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        result = train_gcadti(
            train_df=train,
            val_df=validation,
            test_df=test,
            checkpoint_path=run_dir / "gcadti_best.pt",
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            use_sampling_weight=not args.disable_sampling_weights,
            max_drug_nodes=args.max_drug_nodes,
            max_protein_len=args.max_protein_len,
            early_stopping_patience=args.patience,
            seed=seed,
            history_path=run_dir / "training_history.csv",
            test_metrics_path=run_dir / "test_metrics.json" if test is not None else None,
            test_predictions_path=run_dir / "test_predictions.csv" if test is not None else None,
            checkpoint_metadata_path=run_dir / "checkpoint_metadata.json",
        )
        runs.append(
            {
                "seed": seed,
                "checkpoint": result["checkpoint"],
                "best_epoch": result["best_epoch"],
                "best_val_roc_auc": result["best_val_roc_auc"],
                "test_metrics": result.get("test_metrics"),
            }
        )

    selected = max(runs, key=lambda run: run["best_val_roc_auc"])
    selected_checkpoint = args.output_dir / "gcadti_best.pt"
    shutil.copy2(selected["checkpoint"], selected_checkpoint)
    summary = {
        "selection_rule": "highest validation ROC-AUC across runs; test metrics are not used for selection",
        "selected_seed": selected["seed"],
        "selected_checkpoint": str(selected_checkpoint),
        "runs": runs,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
