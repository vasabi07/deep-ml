import math
def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	# Your code here
	sig_x = 1/(1+math.exp(-x))
	D_sig = sig_x * (1-sig_x)

	tanh_x = math.tanh(x)
	D_tanh = 1 - tanh_x ** 2

	D_relu = 1.0 if x >0 else 0.0

	return {
		"sigmoid": D_sig,
		"tanh": D_tanh,
		"relu": D_relu
	}

