import torch

def linear_forward(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # TODO: implement y = x W^T + b using PyTorch ops
    weights = torch.transpose(W, 0,1)
    y= x @ weights + b
    return y 