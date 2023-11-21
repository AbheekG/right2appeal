import numpy as np
from scipy.stats import uniform, norm
from scipy.integrate import quad
import matplotlib.pyplot as plt

from helper_functions import *


INF = 10
EPS = 1e-5

def appeal_model(sig_A, sig_H, gamma, gamma_A, B):
	
	gamma_H = gamma - gamma_A

	q_A_bar = H_A_inv(1 - gamma_A, sig_A)
	# print(f"q_A_bar = {q_A_bar}")

	A = compute_A(q_A_bar, sig_A)
	# print(f"A = {A}")

	if gamma_H >= min(A,B):
		q_H_bar = -INF
	else:
		q_H_bar = H_H_inv(min(B,A) - gamma_H, q_A_bar, sig_A, sig_H, A, B)

	# print(f"q_H_bar = {q_H_bar}")

	q_exp = expected_q(q_A_bar, q_H_bar, sig_A, sig_H)
	q_exp_A = expected_q_A(q_A_bar, sig_A)
	q_exp_H = expected_q_H(q_A_bar, q_H_bar, sig_A, sig_H)
	# print(f"q_exp = {q_exp}\nq_exp_A = {q_exp_A}\nq_exp_H = {q_exp_H}")

	return q_exp, q_exp_A, q_exp_H


def cost_model(sig_A, sig_H, gamma, gamma_A, B, C):
	
	gamma_H = gamma - gamma_A

	q_A_bar = H_A_inv(1 - gamma_A, sig_A)
	# print(f"q_A_bar = {q_A_bar}")
	
	# compute q_c using C
	q_c = compute_q_c(q_A_bar, sig_A, sig_H, gamma_H, B, C)

	A = compute_A(q_A_bar, sig_A, q_c)

	if gamma_H >= min(A,B):
		q_H_bar = -INF
	else:
		q_H_bar = H_H_inv(min(B,A) - gamma_H, q_A_bar, sig_A, sig_H, A, B, q_c)

	q_exp = expected_q(q_A_bar, q_H_bar, sig_A, sig_H, q_c)
	q_exp_A = expected_q_A(q_A_bar, sig_A)
	q_exp_H = expected_q_H(q_A_bar, q_H_bar, sig_A, sig_H, q_c)

	return q_exp, q_exp_A, q_exp_H


def plot_appeal_model():
	sig_A = 0.2
	sig_H = 0.1
	gamma = 0.1
	B = 0.1

	M = 20
	Gamma_A = np.arange(gamma/M, gamma-gamma/M, gamma/M)
	Q_exp = np.zeros(Gamma_A.size)
	Q_exp_A = np.zeros(Gamma_A.size)
	Q_exp_H = np.zeros(Gamma_A.size)
	
	for i in range(len(Gamma_A)):
		gamma_A = Gamma_A[i]
		Q_exp[i], Q_exp_A[i], Q_exp_H[i] = appeal_model(
			sig_A, sig_H, gamma, gamma_A, B)

	plt.plot(Gamma_A, Q_exp, label='Overall')
	plt.plot(Gamma_A, Q_exp_A, label='AI')
	plt.plot(Gamma_A, Q_exp_H, label="Human")
	plt.legend()
	plt.show()


def plot_cost_model():
	sig_A = 0.2
	sig_H = 0.1
	gamma = 0.1
	B = 0.1
	C = 0.2

	M = 20
	Gamma_A = np.arange(gamma/M, gamma-gamma/M, gamma/M)
	Q_exp = np.zeros(Gamma_A.size)
	Q_exp_A = np.zeros(Gamma_A.size)
	Q_exp_H = np.zeros(Gamma_A.size)
	
	for i in range(len(Gamma_A)):
		gamma_A = Gamma_A[i]
		Q_exp[i], Q_exp_A[i], Q_exp_H[i] = cost_model(
			sig_A, sig_H, gamma, gamma_A, B, C)

	plt.plot(Gamma_A, Q_exp, label='Overall')
	plt.plot(Gamma_A, Q_exp_A, label='AI')
	plt.plot(Gamma_A, Q_exp_H, label="Human")
	plt.legend()
	plt.show()


def f(x):
	"""Unif(0,1) PDF"""
	return uniform.pdf(x)


def F(x):
	"""Unif(0,1) CDF"""
	return uniform.cdf(x)


def g(x, sig):
	"""Norm(0,sig) PDF"""
	return norm.pdf(x, scale=sig)


def G(x, sig):
	"""Norm(0,sig) CDF"""
	return norm.cdf(x, scale=sig)


def h_A(q_A, sig_A):
	"""PDF of q_A seen by AI"""
	return G(1 - q_A, sig_A) - G(-q_A, sig_A)


def H_A(q_A, sig_A):
	"""CDF of q_A seen by AI"""
	def fun(u):
		return G(q_A - u, sig_A)
	return quad(fun, 0, 1)[0]


def H_A_inv(y, sig_A):
	"""inverse of H_A"""
	return inverse_mono(H_A, y, -INF, INF, sig_A)


def a(q, q_c):
	"""appeal probability"""
	if q_c is None:
		return q**2  # change this to experiment with ...
	else:
		# assert q_c >= 0 and q_c < 1
		return float(q >= q_c)


def compute_A(q_A_bar, sig_A, q_c=None):
	"""compute value of A"""
	def fun(q):
		return G(q_A_bar - q, sig_A) * a(q, q_c)
	return quad(fun, 0, 1)[0]


def f_H(q, q_A_bar, sig_A, A, B, q_c=None):
	"""distribution of agents who appeal"""
	if q >= 0 and q <= 1:
		return min(1, B/A) * G(q_A_bar - q, sig_A) * a(q, q_c)
	else:
		return 0


def h_H(q_H, q_A_bar, sig_A, sig_H, A, B, q_c=None):
	"""PDF of q_H seen by Human"""
	def fun(u):
		return G(q_A_bar - u, sig_A) * a(u, q_c) * g(q_H - u, sig_H)
	return min(1, B/A) * quad(fun, 0, 1)[0]


def H_H(q_H, q_A_bar, sig_A, sig_H, A, B, q_c=None):
	"""CDF of q_H seen by Human"""
	def fun(u):
		return G(q_A_bar - u, sig_A) * a(u, q_c) * G(q_H - u, sig_H)
	return min(1, B/A) * quad(fun, 0, 1)[0]


def H_H_inv(y, q_A_bar, sig_A, sig_H, A, B, q_c=None):
	"""inverse of H_H"""
	return inverse_mono(H_H, y, -INF, INF, q_A_bar, sig_A, sig_H, A, B, q_c)


def compute_q_c(q_A_bar, sig_A, sig_H, gamma_H, B, C):
	"""compute q_c the threshold above which agents appeal"""
	def fun(q_c, q_A_bar, sig_A, sig_H, gamma_H, B, C):
		A = compute_A(q_A_bar, sig_A, q_c)
		if gamma_H >= min(A,B):
			q_H_bar = -INF
		else:
			q_H_bar = H_H_inv(min(B,A) - gamma_H, q_A_bar, sig_A, sig_H, A, B, q_c)
		return G(q_c - q_H_bar, sig_H) - C
	return find_root(fun, -10, 10, False, q_A_bar, sig_A, sig_H, gamma_H, B, C)


def expected_q(q_A_bar, q_H_bar, sig_A, sig_H, q_c=None):
	"""expected quality of agent selected"""
	def fun_den(q):
		return (G(q - q_A_bar, sig_A) + 
			G(q_A_bar - q, sig_A) * a(q, q_c) * G(q - q_H_bar, sig_H))
	def fun_num(q):
		return q * fun_den(q)
	return quad(fun_num, 0, 1)[0] / quad(fun_den, 0, 1)[0]


def expected_q_A(q_A_bar, sig_A):
	"""expected quality of agent selected by AI"""
	def fun_den(q):
		return G(q - q_A_bar, sig_A)
	def fun_num(q):
		return q * fun_den(q)
	return quad(fun_num, 0, 1)[0] / quad(fun_den, 0, 1)[0]


def expected_q_H(q_A_bar, q_H_bar, sig_A, sig_H, q_c=None):
	"""expected quality of agent selected by Human"""
	def fun_den(q):
		return G(q_A_bar - q, sig_A) * a(q, q_c) * G(q - q_H_bar, sig_H)
	def fun_num(q):
		return q * fun_den(q)
	return quad(fun_num, 0, 1)[0] / quad(fun_den, 0, 1)[0]


if __name__ == '__main__':
	# plot_appeal_model()
	plot_cost_model()