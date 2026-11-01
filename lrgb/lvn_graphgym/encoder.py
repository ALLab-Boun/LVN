import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_node_encoder


class LVNEmbeddingEncoder(torch.nn.Module):
    """Trainable LVN embeddings, applied to batch.x after the atom/PE node encoder: the i-th
    LVN of every group gets e_i added to (add) or in place of (replace) the representation it
    inherits from its central node."""

    def __init__(self, dim_emb):
        super().__init__()
        self.emb = torch.nn.Embedding(cfg.lvn.n_c, dim_emb)
        torch.nn.init.xavier_uniform_(self.emb.weight)
        self.method = cfg.lvn.embedding_method

    def forward(self, batch):
        mask = batch.virtual_mask
        values = self.emb(batch.virtual_node_local_indices[mask])
        x = batch.x.clone()
        x[mask] = x[mask] + values if self.method == "add" else values
        batch.x = x
        return batch


register_node_encoder("lvn", LVNEmbeddingEncoder)
