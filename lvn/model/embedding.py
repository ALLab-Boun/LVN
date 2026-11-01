import torch


class LVNEmbedding(torch.nn.Module):
    """Trainable LVN embeddings e_1, ..., e_{n_c}, shared across LVN groups (Section 4.3).

    The i-th LVN of every group gets e_i: added to the representation it inherits from its
    central node ("add") or used instead of it ("replace")."""

    def __init__(self, d_model, n_c, method="add", init_std=None):
        super().__init__()
        if method not in ("add", "replace"):
            raise ValueError(f"Unknown LVN embedding method: {method}")
        self.emb = torch.nn.Embedding(n_c, d_model)  # N(0, 1) initialisation
        if init_std is not None:
            torch.nn.init.normal_(self.emb.weight, std=init_std)
        self.method = method

    def forward(self, x, local_indices, virtual_mask):
        x = x.clone()
        values = self.emb(local_indices[virtual_mask])
        if self.method == "add":
            x[virtual_mask] = x[virtual_mask] + values
        else:
            x[virtual_mask] = values
        return x
