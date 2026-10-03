from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from dgllife.model.gnn.gcn import GCN
except ImportError:  # pragma: no cover
    GCN = None


class Conv1dReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class LinearReLU(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class GCNReLU(nn.Module):
    def __init__(self, in_feats: int, hidden_feats: list[int]):
        super().__init__()
        if GCN is None:
            raise RuntimeError("dgllife is required for GCADTI")
        self.gnn = GCN(in_feats, hidden_feats, gnn_norm=None, activation=None)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop("h")
        batch_size = batch_graph.batch_size
        node_feats = self.gnn(batch_graph, node_feats)
        return node_feats.view(batch_size, -1, self.output_feats)


class AtomProjection(nn.Module):
    """Per-atom feature projection used by the no-GCN ablation.

    The graph topology is not accessed; atom features are projected independently
    so that the downstream pooling and attention dimensions remain unchanged.
    """

    def __init__(self, in_feats: int, out_feats: int):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(in_feats, out_feats), nn.ReLU())

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop("h")
        batch_size = batch_graph.batch_size
        node_feats = self.projection(node_feats)
        return node_feats.view(batch_size, -1, node_feats.shape[-1])


class BilinearPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, c_m: int, c_n: int):
        super().__init__()
        self.conv_a = nn.Conv1d(in_channels, c_m, kernel_size=1)
        self.conv_b = nn.Conv1d(in_channels, c_n, kernel_size=1)
        self.linear = nn.Linear(c_m, out_channels)

    def forward(self, x):
        a = self.conv_a(x)
        b = self.conv_b(x)
        att_maps = F.softmax(b, dim=-1)
        global_descriptors = torch.bmm(a, att_maps.permute(0, 2, 1))
        global_descriptor = torch.mean(global_descriptors, dim=-1)
        return self.linear(global_descriptor).unsqueeze(1)


class MutualAttention(nn.Module):
    def __init__(self, in_channels: int, att_size: int, c_m: int, c_n: int):
        super().__init__()
        self.bipool = BilinearPooling(in_channels, in_channels, c_m, c_n)
        self.linear_source = nn.Linear(in_channels, att_size)
        self.linear_target = nn.Linear(in_channels, att_size)

    def forward(self, source, target):
        descriptor = self.linear_source(self.bipool(source))
        target_org = target
        target = self.linear_target(target.permute(0, 2, 1)).permute(0, 2, 1)
        att_maps = torch.sigmoid(torch.bmm(descriptor, target))
        return F.relu(target_org + target_org * att_maps)


class GCADTI(nn.Module):
    def __init__(self, dropout: float = 0.3, interaction_mode: str = "mutual"):
        super().__init__()
        valid_modes = {
            "mutual",
            "concat",
            "drug_to_protein",
            "protein_to_drug",
            "no_gcn",
            "no_conv1d",
        }
        if interaction_mode not in valid_modes:
            raise ValueError(f"interaction_mode must be one of {sorted(valid_modes)}")
        self.interaction_mode = interaction_mode
        self.prot_embed = nn.Embedding(26, 128, padding_idx=0)
        self.prot_conv1 = Conv1dReLU(128, 128, 3)
        self.bn1 = nn.BatchNorm1d(128)
        self.prot_conv2 = Conv1dReLU(128, 128, 6)
        self.bn2 = nn.BatchNorm1d(128)
        self.prot_conv3 = Conv1dReLU(128, 128, 9)
        self.bn3 = nn.BatchNorm1d(128)
        self.prot_pool = nn.AdaptiveMaxPool1d(1)

        self.drug_gcn = GCNReLU(75, [128, 128, 128])
        self.drug_atom_projection = AtomProjection(75, 128)
        self.drug_pool = nn.AdaptiveMaxPool1d(1)

        if interaction_mode in {"mutual", "drug_to_protein", "protein_to_drug"}:
            self.prot_mut_att = MutualAttention(128, 128, 128, 32)
            self.drug_mut_att = MutualAttention(128, 128, 128, 32)

        self.linear1 = LinearReLU(256, 1024)
        self.drop1 = nn.Dropout(dropout)
        self.linear2 = LinearReLU(1024, 1024)
        self.drop2 = nn.Dropout(dropout)
        self.linear3 = LinearReLU(1024, 512)
        self.drop3 = nn.Dropout(dropout)
        self.out_layer = nn.Linear(512, 1)

    def forward(self, prot_x, drug_x):
        prot_x = self.prot_embed(prot_x).permute(0, 2, 1)
        if self.interaction_mode != "no_conv1d":
            prot_x = self.bn1(self.prot_conv1(prot_x))
            prot_x = self.bn2(self.prot_conv2(prot_x))
            prot_x = self.bn3(self.prot_conv3(prot_x))

        if self.interaction_mode == "no_gcn":
            drug_x = self.drug_atom_projection(drug_x)
        else:
            drug_x = self.drug_gcn(drug_x)

        if self.interaction_mode == "mutual":
            prot_x_g = self.prot_mut_att(drug_x.permute(0, 2, 1), prot_x)
            drug_x_g = self.drug_mut_att(prot_x, drug_x.permute(0, 2, 1))
        elif self.interaction_mode == "drug_to_protein":
            prot_x_g = self.prot_mut_att(drug_x.permute(0, 2, 1), prot_x)
            drug_x_g = drug_x.permute(0, 2, 1)
        elif self.interaction_mode == "protein_to_drug":
            prot_x_g = prot_x
            drug_x_g = self.drug_mut_att(prot_x, drug_x.permute(0, 2, 1))
        else:
            # Ablation: retain the encoders and the same pooling/predictor,
            # but remove the bidirectional conditional interaction module.
            prot_x_g = prot_x
            drug_x_g = drug_x.permute(0, 2, 1)

        prot_vec = self.prot_pool(prot_x_g).squeeze(-1)
        drug_vec = self.drug_pool(drug_x_g).squeeze(-1)
        x = torch.cat([prot_vec, drug_vec], dim=-1)
        x = self.drop1(self.linear1(x))
        x = self.drop2(self.linear2(x))
        x = self.drop3(self.linear3(x))
        return prot_vec, self.out_layer(x)
