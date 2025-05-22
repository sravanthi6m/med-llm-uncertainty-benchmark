import numpy as np
import torch
import torch.nn.functional as F

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def log_softmax(logits: torch.Tensor) -> torch.Tensor:
    logits = logits - torch.max(logits)
    return F.log_softmax(logits, dim=0)

