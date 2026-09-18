import torch
from typing import List
import torch.nn.functional as F
def log_softmax(scores: List[float]) -> torch.Tensor:
    """
    Compute the log-softmax of a 1D list of scores using PyTorch.
    Args:
        scores: list of floats
    Returns:
        torch.Tensor of log-softmax values
    """
    # Your code here

    z_tensor = torch.tensor(scores,dtype=torch.float32)

    return F.log_softmax(z_tensor, dim=0)
