import torch


@torch.no_grad()
def accuracy(pred: torch.Tensor, y: torch.Tensor):
    if pred.ndim > 1:
        _, pred_cls = torch.max(pred, dim=1)
    else:
        pred_cls = pred > 0.5
    correct = (pred_cls == y).sum()
    accuracy = (correct / len(y)) * 100.0
    accuracy = accuracy.item()
    return accuracy
