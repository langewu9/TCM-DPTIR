from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

try:
    import dgl
    from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph
except ImportError:  # pragma: no cover
    dgl = None
    CanonicalAtomFeaturizer = None
    CanonicalBondFeaturizer = None
    smiles_to_bigraph = None


VOCAB_PROTEIN = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7, "I": 8,
    "H": 9, "K": 10, "M": 11, "L": 12, "O": 13, "N": 14, "Q": 15,
    "P": 16, "S": 17, "R": 18, "U": 19, "T": 20, "W": 21, "V": 22,
    "Y": 23, "X": 24, "Z": 25,
}


def encode_protein(sequence: str, max_len: int = 1200) -> np.ndarray:
    encoded = [VOCAB_PROTEIN.get(aa.upper(), VOCAB_PROTEIN["X"]) for aa in str(sequence)]
    if len(encoded) < max_len:
        encoded = encoded + [0] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]
    return np.asarray(encoded, dtype=np.int64)


def graph_collate_func(batch):
    drugs, proteins, labels = zip(*batch)
    drugs = dgl.batch(drugs)
    return (
        drugs,
        torch.tensor(np.asarray(proteins), dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )


class GCADTIDataset(data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        max_drug_nodes: int = 290,
        max_protein_len: int = 1200,
        cache_encodings: bool = True,
    ):
        if dgl is None:
            raise RuntimeError("dgl and dgllife are required for GCADTIDataset")
        self.df = df.reset_index(drop=True)
        self.max_drug_nodes = max_drug_nodes
        self.max_protein_len = max_protein_len
        self.cache_encodings = cache_encodings
        # Full-TCM training contains many repeated compounds and targets.  Cache
        # their deterministic graph/sequence encodings so later epochs do not
        # repeatedly invoke RDKit and rebuild identical padded tensors.
        self._graph_cache = {}
        self._protein_cache = {}
        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.smiles_to_graph = partial(smiles_to_bigraph, add_self_loop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        smiles = str(row["smiles"])
        graph = self._graph_cache.get(smiles) if self.cache_encodings else None
        if graph is None:
            graph = self.smiles_to_graph(
                smiles=smiles,
                node_featurizer=self.atom_featurizer,
                edge_featurizer=self.bond_featurizer,
            )
            node_feats = graph.ndata.pop("h")
            num_actual_nodes = node_feats.shape[0]
            if num_actual_nodes > self.max_drug_nodes:
                raise ValueError(
                    f"SMILES at row {index} has {num_actual_nodes} atoms, "
                    f"which exceeds max_drug_nodes={self.max_drug_nodes}. "
                    "Increase model.max_drug_nodes in config/stroke_dti.yaml or pass "
                    "--max-drug-nodes to the training/prediction script."
                )
            num_virtual_nodes = max(self.max_drug_nodes - num_actual_nodes, 0)
            virtual_bit = torch.zeros([node_feats.shape[0], 1])
            graph.ndata["h"] = torch.cat((node_feats, virtual_bit), dim=1)
            if num_virtual_nodes:
                virtual_node_feat = torch.cat(
                    (torch.zeros(num_virtual_nodes, 74), torch.ones(num_virtual_nodes, 1)),
                    dim=1,
                )
                graph.add_nodes(num_virtual_nodes, {"h": virtual_node_feat})
            graph = graph.add_self_loop()
            if self.cache_encodings:
                self._graph_cache[smiles] = graph

        sequence = str(row["sequence"])
        protein = self._protein_cache.get(sequence) if self.cache_encodings else None
        if protein is None:
            protein = encode_protein(sequence, self.max_protein_len)
            if self.cache_encodings:
                self._protein_cache[sequence] = protein
        return graph, protein, int(row["label"])
