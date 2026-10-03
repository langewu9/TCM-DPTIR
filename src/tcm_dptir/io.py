from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def normalize_dti_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "SMILES": "smiles",
        "Smiles": "smiles",
        "canonical_smiles": "smiles",
        "Target": "accession",
        "ACCESSION": "accession",
        "Label": "label",
    }
    out = df.rename(columns={k: v for k, v in rename.items() if k in df.columns}).copy()
    required = {"smiles", "accession", "label"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing required DTI columns: {sorted(missing)}")
    out = out.dropna(subset=["smiles", "accession", "label"])
    out["label"] = out["label"].astype(int)
    return out
