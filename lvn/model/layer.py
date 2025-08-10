import math

import torch


class GroupPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, group_size):
        super(GroupPositionalEncoding, self).__init__()

        pe = torch.zeros(group_size, d_model)
        position = torch.arange(0, group_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x, local_indices, group_mask):
        # x: [N, D], PE: [k, D], groups: [N]
        x_0 = x
        x_0[group_mask] = self.pe[local_indices[group_mask]]
        return x_0


class TrainableGroupPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, group_size, method="add"):
        super(TrainableGroupPositionalEncoding, self).__init__()
        self.group_size = group_size
        self.num_embeddings = group_size - 1 if method == "add_one" else group_size
        self.pe = torch.nn.Embedding(self.num_embeddings, d_model)
        self.method = method

    def forward(self, x, local_indices, group_mask):
        # x: [N, D], PE: [k, D], local_indices: [N], group_mask: [N]
        if self.method == "add":
            pe_values = self.pe(local_indices[group_mask])
            x[group_mask] = x[group_mask] + pe_values
        elif self.method == "replace":
            pe_values = self.pe(local_indices[group_mask])
            x[group_mask] = pe_values
        elif self.method == "add_one":
            actual_mask = group_mask & (local_indices > 0)
            pe_values = self.pe(local_indices[actual_mask] - 1)
            x[actual_mask] = x[actual_mask] + pe_values

        return x


class TrainableSeparateGroupPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_groups, group_size, method="add"):
        super(TrainableSeparateGroupPositionalEncoding, self).__init__()
        self.group_size = group_size
        self.num_embeddings = (
            max_groups * (group_size - 1)
            if method == "add_one"
            else max_groups * group_size
        )
        self.pe = torch.nn.Embedding(self.num_embeddings, d_model)
        self.max_groups = max_groups
        self.method = method

    def forward(self, x, local_indices, group_mask, batch_indicator):
        if self.method == "add" or self.method == "replace":
            actual_mask = group_mask
        elif self.method == "add_one":
            actual_mask = group_mask & (local_indices > 0)

        pe_indices = torch.cat(
            [
                torch.arange(count, device=x.device)
                for count in torch.unique_consecutive(
                    batch_indicator[actual_mask], return_counts=True
                )[1]
            ]
        )
        pe_values = self.pe(pe_indices)

        if self.method == "add" or self.method == "add_one":
            x[actual_mask] = x[actual_mask] + pe_values
        elif self.method == "replace":
            x[actual_mask] = pe_values

        return x
