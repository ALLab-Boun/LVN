import torch

from .locality import distance_distortion, lvn_locality, lvn_node_mapping


@torch.no_grad()
def accuracy(pred, y):
    pred_cls = pred.argmax(1) if pred.ndim > 1 else pred > 0
    return (pred_cls == y).float().mean().item() * 100.0


@torch.no_grad()
def evaluate_loaders(model, loaders, loss_func=None):
    """Accuracy (and loss) of a graph-level model on each loader."""
    model.eval()
    device = next(model.parameters()).device
    out = {}
    for name, loader in loaders.items():
        preds, ys = [], []
        for batch in loader:
            batch = batch.to(device)
            preds.append(model(batch, pool="root_mask" not in batch).squeeze(1))
            ys.append(batch.y)
        pred, y = torch.cat(preds), torch.cat(ys)
        out[f"{name}_eval"] = accuracy(pred, y)
        if loss_func is not None:
            out[f"{name}_loss"] = loss_func(pred, y).item()
    return out


@torch.no_grad()
def evaluate_masks(model, graph, masks, loss_func=None):
    """Accuracy (and loss) of a node-level model on each mask of its output rows."""
    model.eval()
    pred = model(graph, pool=False)
    if pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    out = {}
    for name, mask in masks.items():
        out[f"{name}_eval"] = accuracy(pred[mask], graph.y[mask])
        if loss_func is not None:
            out[f"{name}_loss"] = loss_func(pred[mask], graph.y[mask]).item()
    return out


__all__ = ["accuracy", "evaluate_loaders", "evaluate_masks", "distance_distortion", "lvn_locality", "lvn_node_mapping"]
