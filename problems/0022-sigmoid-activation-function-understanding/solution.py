import torch

def sigmoid(z: float) -> float:
    """
    Compute the sigmoid activation function.
    Input:
      - z: float or torch scalar tensor
    Returns:
      - sigmoid(z) as Python float rounded to 4 decimals.
    """
    # Your implementation here
    z_tensor = torch.tensor(z,dtype = torch.float32)
    sigmoid_val = torch.sigmoid(z_tensor)
    rounded_num = round(sigmoid_val.item(),4)
    return rounded_num
    pass
