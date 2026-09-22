import torch

def bce_with_logits(logits, targets):
    """Mean BCE-with-logits loss, numerically stable, rounded to 4 decimals.

    Args:
        logits (torch.Tensor): 1-D raw logits.
        targets (torch.Tensor): 1-D binary targets in {0, 1}, same shape.

    Returns:
        float: mean loss rounded to 4 decimal places.
    """
    # TODO: stable BCE-with-logits, mean, round to 4 decimals
    x = logits
    y= targets

    t1 = torch.clamp(x,min=0)
    t2 = x*y
    t3 = torch.log(1+ torch.exp(-torch.abs(x)))
    loss = torch.mean(t1-t2+t3)
    return round(loss.item(),4)
