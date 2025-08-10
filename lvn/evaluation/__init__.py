import torch

from .metrics import accuracy
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def inference(model, loader):
    preds = []
    ys = []
    model.eval()
    for batch in loader:
        pred = model(batch, pool=False if "root_mask" in batch else True)
        pred = pred.squeeze(1)
        preds.append(pred)
        ys.append(batch.y)

    return torch.cat(preds), torch.cat(ys)


@torch.no_grad()
def get_2d_node_embeddings(model, graph):
    embeddings = model(graph, pooling=None, embed=True)
    embeddings_np = embeddings.cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_np)

    return embeddings_2d


@torch.no_grad()
def evaluate(
    model,
    loaders_dict=None,
    dataset=None,
    split=0,
    loss_func=None,
    eval_metric="acc",
):
    if loaders_dict is not None:
        eval_dict = {
            "eval_metric": eval_metric,
        }
        for loader_name, loader in loaders_dict.items():
            pred, y = inference(model, loader)
            if eval_metric == "acc":
                eval = accuracy(pred, y)
            else:
                eval = None

            if loss_func:
                loss = loss_func(pred, y).detach().item()
            else:
                loss = None

            eval_dict[f"{loader_name}_loss"] = round(loss, 3) if loss_func else None
            eval_dict[f"{loader_name}_eval"] = round(eval, 3)

    elif dataset is not None:
        graph = dataset[0]
        y = graph.y
        if graph.train_mask.ndim > 1:
            train_mask = graph.train_mask[:, split]
            val_mask = graph.val_mask[:, split]
            test_mask = graph.test_mask[:, split]
        else:
            train_mask = graph.train_mask
            val_mask = graph.val_mask
            test_mask = graph.test_mask

        model.eval()
        pred = model(graph, pool=False)

        if loss_func:
            if pred.shape[-1] == 1:
                pred = pred.squeeze()
            train_loss = loss_func(pred[train_mask], y[train_mask]).detach().item()
            val_loss = loss_func(pred[val_mask], y[val_mask]).detach().item()
            test_loss = loss_func(pred[test_mask], y[test_mask]).detach().item()
        else:
            train_loss = None
            val_loss = None
            test_loss = None

        if eval_metric == "acc":
            if pred.shape[-1] == 1:
                pred_cls = pred >= 0.5
            else:
                pred_cls = pred.argmax(1)
            train_eval = (
                pred_cls[train_mask] == y[train_mask]
            ).float().mean().item() * 100.0
            val_eval = (pred_cls[val_mask] == y[val_mask]).float().mean().item() * 100.0
            test_eval = (
                pred_cls[test_mask] == y[test_mask]
            ).float().mean().item() * 100.0
        elif eval_metric == "auc":
            train_eval = (
                roc_auc_score(y[train_mask].cpu(), pred[train_mask].cpu()) * 100.0
            )
            val_eval = roc_auc_score(y[val_mask].cpu(), pred[val_mask].cpu()) * 100.0
            test_eval = roc_auc_score(y[test_mask].cpu(), pred[test_mask].cpu()) * 100.0

        eval_dict = {
            "train_loss": round(train_loss, 3) if loss_func else None,
            "train_eval": round(train_eval, 3),
            "test_loss": round(test_loss, 3) if loss_func else None,
            "test_eval": round(test_eval, 3),
            "val_loss": round(val_loss, 3) if loss_func else None,
            "val_eval": round(val_eval, 3),
            "eval_metric": eval_metric,
        }

    else:
        raise ValueError("Either loader or dataset must be provided")

    return eval_dict


__all__ = ["inference", "accuracy"]
