import subprocess
import time
from functools import wraps
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from ..logging import logger


def auxiliary_virtual_emb_loss(virtual_emb):
    normalized_emb = F.normalize(virtual_emb, dim=1)
    similarity_matrix = normalized_emb @ normalized_emb.T
    mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)
    similarity_matrix = similarity_matrix * (1 - mask)
    loss = (similarity_matrix**2).mean()

    return loss


def plot_grads(history):
    plt.figure(figsize=(10, 6))
    for name, norms in history.items():
        plt.plot(norms, label=name)
    plt.yscale("log")
    plt.xlabel("Steps")
    plt.ylabel("Gradient Norm")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("grads.png")
    plt.close()


def register_gradient_hooks(model):
    gradient_norms = {}

    def hook_fn(name):
        def fn(grad):
            gradient_norms[name] = grad.norm().item()

        return fn

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(hook_fn(name))
    return gradient_norms


def gpu_memory_usage():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        capture_output=True,
        text=True,
    )
    used_memory = result.stdout.strip()
    used_memory = used_memory.split("\n")[0]
    used_memory = int(used_memory)
    return used_memory / 1024


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        logger.info(
            f"function '{func.__name__}' executed in {elapsed_time:.1f} minutes"
        )
        return result

    return wrapper
