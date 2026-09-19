import torch
import torch.nn as nn

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """
    # Your code here
    x = features
    weights = initial_weights.clone().requires_grad_(True)
    bias = torch.tensor(initial_bias,requires_grad = True)
    mse_list = []
    for i in range(epochs):
        y_hat = torch.matmul(x, weights) + bias
        z = torch.sigmoid(y_hat)
        mse_func = nn.MSELoss()
        mse = mse_func(z, labels)
        mse_list.append(round(mse.item(),4))
        if weights.grad is not None:
            weights.grad.zero_()
        if bias.grad is not None:
            bias.grad.zero_()
        mse.backward()
        with torch.no_grad():
            weights -= learning_rate * weights.grad
            bias -= learning_rate * bias.grad
    return [round(w, 4) for w in weights.tolist()], round(bias.item(), 4), mse_list






