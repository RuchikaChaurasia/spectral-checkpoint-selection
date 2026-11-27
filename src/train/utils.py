"""Helpers shared by the training scripts (owner: Disha)."""
from __future__ import annotations

import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from src.data.officehome import REPO_ROOT

HEAD_KEYS = ("fc.weight", "fc.bias")


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Give every DataLoader worker its own seed derived from the loader's generator."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def build_model(num_classes: int, mode: str, weights: str = "IMAGENET1K_V1") -> nn.Module:
    """ImageNet-pretrained ResNet-18 with a new 65-way head.

    mode='linear_probe' freezes everything except the head.
    weights='none' (offline testing only) skips the pretrained weights.
    """
    w = None if weights.lower() == "none" else ResNet18_Weights[weights]
    model = resnet18(weights=w)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # initialised from the seeded RNG
    if mode == "linear_probe":
        for name, p in model.named_parameters():
            p.requires_grad = name in HEAD_KEYS
    return model


def checkpoint_state(model: nn.Module, mode: str) -> dict:
    """Weights to save each epoch: the whole model for fine-tuning, only the head for the probe."""
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if mode == "linear_probe":
        sd = {k: sd[k] for k in HEAD_KEYS}
    return sd


def load_checkpoint(model: nn.Module, path: str | Path, mode: str) -> nn.Module:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if mode == "linear_probe":
        missing, unexpected = model.load_state_dict(sd, strict=False)
        bad = [k for k in missing if k in HEAD_KEYS] + list(unexpected)
        if bad:
            raise RuntimeError(f"Bad linear-probe checkpoint {path}: {bad}")
    else:
        model.load_state_dict(sd)
    return model


def env_info() -> dict:
    import datasets
    import torchvision

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "datasets": datasets.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        if torch.cuda.is_available() else None,
        "cpu_count": os.cpu_count(),
        "hostname": platform.node(),
    }
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip()
        info["git_dirty"] = bool(dirty)
    except Exception:  # noqa: BLE001 - not a git checkout
        info["git_commit"] = None
    return info


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0))
