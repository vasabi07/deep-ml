import torch

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a 2x2 matrix using PyTorch.
    Input: 2x2 tensor; Output: 1-D tensor with the two eigenvalues in descending order (highest to lowest).
    """
    # Your implementation here
    vals = torch.linalg.eigvals(matrix)
    vals_real = vals.real
    vals_sorted, _ = torch.sort(vals_real, descending=True)
    return vals_sorted
