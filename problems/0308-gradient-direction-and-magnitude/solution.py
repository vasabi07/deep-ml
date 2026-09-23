import torch

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient (float)
		- direction: Unit vector (torch.Tensor) in direction of steepest ascent
		- descent_direction: Unit vector (torch.Tensor) in direction of steepest descent
	"""
	# Your code here

	g = torch.tensor(gradient,dtype = torch.float32)
	magnitude = torch.norm(g)

	if magnitude == 0:
        zero_vec = torch.zeros_like(g)
        return {
            'magnitude': 0.0,
            'direction': zero_vec,
            'descent_direction': zero_vec
        }

	direction = g/magnitude
	descent_direction = -direction

	return {
		'magnitude': magnitude.item(),
        'direction': direction,
        'descent_direction': descent_direction
	}




