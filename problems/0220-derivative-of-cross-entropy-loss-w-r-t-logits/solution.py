import torch
import torch.nn.functional as F

def cross_entropy_derivative(logits: torch.Tensor, target: int) -> torch.Tensor:
    """
    Compute the derivative of cross-entropy loss with respect to logits.
    
    Args:
        logits: Raw model outputs tensor
        target: Index of the true class
        
    Returns:
        Gradient tensor
    """
    # Your code here - can use autograd or the analytical formula
    logits_t = logits.clone().requires_grad_(True)
    target_t = torch.tensor(target,dtype=torch.long)
    Loss = F.cross_entropy(logits_t, target_t)
    return torch.autograd.grad(Loss,logits_t)[0]