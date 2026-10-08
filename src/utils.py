"""Small helper functions shared by all experiments."""
import json
import os
import random

import numpy as np
import torch


def set_seed(seed=0):
    """Makes runs repeatable: same seed -> same random points and weights."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    """Uses the GPU if one is available (e.g. on Google Colab), else the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(array, device, requires_grad=False):
    """NumPy array -> float32 PyTorch tensor on the chosen device."""
    t = torch.tensor(np.asarray(array), dtype=torch.float32, device=device)
    if requires_grad:
        t.requires_grad_(True)
    return t


def grad(y, x):
    """
    Derivative dy/dx using automatic differentiation (autograd).
    This is the key PINN ingredient: it gives exact derivatives of the
    network output with respect to its inputs, so no mesh is needed.
    create_graph=True lets us take derivatives of derivatives (e.g. p_xx).
    """
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                               create_graph=True)[0]


def predict(model, X_np, device):
    """Evaluate the trained network on a NumPy array of points."""
    with torch.no_grad():
        return model(to_tensor(X_np, device)).cpu().numpy()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def relative_l2(pred, ref):
    """||pred - ref|| / ||ref||  (standard PINN accuracy metric)."""
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))
