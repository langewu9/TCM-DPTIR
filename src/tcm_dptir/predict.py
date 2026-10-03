from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from tcm_dptir.datasets import GCADTIDataset, graph_collate_func
from tcm_dptir.io import write_csv
from tcm_dptir.models.gcadti import GCADTI


def predict_pairs(
    pair_df: pd.DataFrame,
    checkpoint_path: Path,
    output_path: Path,
    device: str = "cpu",
    batch_size: int = 64,
    max_drug_nodes: int = 290,
    max_protein_len: int = 1200,
) -> pd.DataFrame:
    torch_device = torch.device(device)
    model = GCADTI().to(torch_device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch_device))
    model.eval()

    scoring_df = pair_df.copy()
    if "label" not in scoring_df.columns:
        scoring_df["label"] = 0
    dataset = GCADTIDataset(
        scoring_df,
        max_drug_nodes=max_drug_nodes,
        max_protein_len=max_protein_len,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=graph_collate_func)

    scores: list[float] = []
    with torch.no_grad():
        for drug, protein, _ in loader:
            drug = drug.to(torch_device)
            protein = protein.to(torch_device)
            _, logits = model(protein, drug)
            scores.extend(torch.sigmoid(logits.squeeze(1)).cpu().numpy().tolist())

    out = pair_df.copy()
    out["gcadti_score"] = scores
    write_csv(out, output_path)
    return out
