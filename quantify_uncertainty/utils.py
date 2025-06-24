import numpy as np
import torch
import torch.nn.functional as F
import json, random
from sklearn.model_selection import train_test_split


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def log_softmax(logits: torch.Tensor) -> torch.Tensor:
    logits = logits - torch.max(logits)
    return F.log_softmax(logits, dim=0)


def load_and_split_jsonl(path, cal_ratio=0.5, seed=42):
    """Returns cal_split, test_split (each is a list of dicts)."""
    with open(path) as f:
        data = [json.loads(l) for l in f]
    return train_test_split(data, train_size=cal_ratio, random_state=seed)
