from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tcm_dptir.predict import predict_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score compound-target pairs with a GCADTI checkpoint")
    parser.add_argument("--pairs", required=True, type=Path, help="CSV with smiles and sequence columns")
    parser.add_argument("--checkpoint", required=True, type=Path, help="User-supplied model state dictionary")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-drug-nodes", type=int, default=64)
    parser.add_argument("--max-protein-len", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = pd.read_csv(args.pairs)
    missing = {"smiles", "sequence"} - set(pairs.columns)
    if missing:
        raise ValueError(f"Input table is missing required columns: {sorted(missing)}")
    predict_pairs(
        pair_df=pairs,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device=args.device,
        batch_size=args.batch_size,
        max_drug_nodes=args.max_drug_nodes,
        max_protein_len=args.max_protein_len,
    )
    print(f"Scored {len(pairs)} pairs -> {args.output}")


if __name__ == "__main__":
    main()
