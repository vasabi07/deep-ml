import torch
import numpy as np


def fit_linear_regression(X, y, lr=0.1, steps=500):
    """Fit y ~= X @ w + b with full-batch GD using only autograd.

    Args:
        X: Float tensor (N, D)
        y: Float tensor (N,) or (N, 1)
        lr: learning rate
        steps: number of gradient descent iterations

    Returns:
        w: Float tensor (D,) learned weights (no grad)
        b: Float tensor scalar learned bias (no grad)
    """
    # TODO: ensure y is shape (N,)
    # TODO: initialize w (D,) and b with requires_grad=True
    # TODO: for each step:
    #   - predict, compute MSE loss
    #   - loss.backward()
    #   - manual GD update under torch.no_grad()
    #   - zero gradients
    # TODO: return detached w, b
    D = X.shape[1]
    W = torch.zeros(D,dtype = torch.float32,requires_grad=True)
    b = torch.zeros(1,dtype = torch.float32,requires_grad=True)
    Y = y.reshape(-1)
    for i in range(steps):
        y_pred = X @ W + b
        Loss = torch.mean((y_pred - Y)**2)
        Loss.backward()
        with torch.no_grad():
            W -= lr*W.grad
            b -= lr * b.grad
        
        W.grad.zero_()
        b.grad.zero_()
    return W.detach(),b.detach()
    raise NotImplementedError
