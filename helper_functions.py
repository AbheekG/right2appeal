def inverse_mono(fun, y, lb, ub, *args):
	"""
	computes the inverse of a monotone function fun at y
	lb: lower bound
	ub: upper bound
	"""
	def _fun(x, *args):
		return fun(x, *args) - y

	return find_root(_fun, lb, ub, False, *args)


def find_root(fun, lb, ub, prin, *args):
	"""
	computes the root of a monotone function fun
	lb: lower bound
	ub: upper bound
	"""
	assert fun(lb, *args) <= 0, f"fun value should be below 0 at {lb}"
	assert fun(ub, *args) >= 0, f"fun value should be above 0 at {ub}"

	eps = 1e-5
	while (abs(lb-ub) > eps):
		x = (lb + ub)/2
		if prin:
			print(fun(x, *args), x)
		if fun(x, *args) <= 0:
			lb = x
		else:
			ub = x

	return lb