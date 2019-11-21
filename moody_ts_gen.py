# Generates timeseries by random walk with autoregresive trend a la Moody et al 2001

import numpy as np


def generate_timeseries(T, alpha=0.9, k=3):
	T=10000
	alpha=0.9
	k=3

	eps = np.random.randn(T)
	v = np.random.randn(T)
	z = np.zeros(T)
	p = np.zeros(T)
	beta = np.zeros(T)
	p[0] = 1


	for t in range(1, T):
	    p[t] = p[t-1] + beta[t-1] + k*eps[t]
	    beta[t] = alpha * beta[t-1] + v[t]

	#     R = np.max(p[:t+1]) - np.min(p[:t+1])
	#     z[t] = np.exp(p[t] / R)
	R = np.max(p) - np.min(p)
	z = np.exp(p / R)

	return z

