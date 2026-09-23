import torch

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> torch.Tensor:
    """
    Compute partial derivatives of multivariable functions using PyTorch autograd.
    
    Args:
        func_name: Function identifier
            'poly2d': f(x,y) = x^2*y + x*y^2
            'exp_sum': f(x,y) = e^(x+y)
            'product_sin': f(x,y) = x*sin(y)
            'poly3d': f(x,y,z) = x^2*y + y*z^2
            'squared_error': f(x,y) = (x-y)^2
        point: Point (x, y) or (x, y, z) at which to evaluate
    
    Returns:
        Tensor of partial derivatives [df/dx, df/dy, ...] at point
    """
    # Your code here
    x_tensor = torch.tensor(point,dtype = torch.float32,requires_grad = True)

    if func_name == 'poly2d':
        x, y = x_tensor[0], x_tensor[1]
        f = x**2 * y + x * y**2
    elif func_name == 'exp_sum':
        x, y = x_tensor[0], x_tensor[1]
        f = torch.exp(x + y)
    elif func_name == 'product_sin':
        x, y = x_tensor[0], x_tensor[1]
        f = x * torch.sin(y)
    elif func_name == 'poly3d':
        x, y, z = x_tensor[0], x_tensor[1], x_tensor[2]
        f = x**2 * y + y * z**2
    elif func_name == 'squared_error':
        x, y = x_tensor[0], x_tensor[1]
        f = (x - y)**2

    f.backward()
    return x_tensor.grad


