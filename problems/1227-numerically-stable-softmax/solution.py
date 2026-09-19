import torch

def softmax(t, dim):
    """Numerically stable softmax along dim.

    Args:
        t (torch.Tensor): input tensor
        dim (int): dimension along which to apply softmax

    Returns:
        torch.Tensor: tensor of same shape as t; slices along dim sum to 1
    """
    # TODO: subtract max along dim, exp, then normalize
    max_vals = torch.max(t,dim=dim,keepdim=True).values
    new_t = torch.sub(t,max_vals)
    exp_new_t = torch.exp(new_t)
    sum_exp_t = torch.sum(exp_new_t,dim=dim,keepdim=True)
    return exp_new_t/sum_exp_t