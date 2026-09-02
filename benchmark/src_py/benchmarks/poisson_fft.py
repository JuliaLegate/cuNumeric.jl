import math

import cupynumeric as np
import numpy as onp

from core import register_benchmark, rand_array


def _integer_fftfreq(n):
    n2 = n // 2
    if n % 2 == 0:
        return list(range(0, n2)) + list(range(-n2, 0))
    return list(range(0, n2 + 1)) + list(range(-n2, 0))


def _poisson_inv_laplacian(T, n):
    freq = onp.array(_integer_fftfreq(n), dtype=T)
    fx = freq[:, None]
    fy = freq[None, :]
    k2 = T(4.0 * math.pi**2) * (fx * fx + fy * fy)
    invk = onp.zeros((n, n), dtype=T)
    onp.divide(-1.0, k2, out=invk, where=k2 != 0)
    return invk


class PoissonFFT:
    name = "poisson_fft"

    def __init__(self, T, N, M):
        self.T, self.N, self.M = T, N, M

    def dims(self):
        return self.N, self.M

    def total_flops(self):
        # Same breakdown as benchmark/src/benchmarks/poisson_fft.jl
        n = self.N
        m = self.M
        n2 = n * n
        return m * (20 * n2 * math.log2(n) + 6 * n2)

    def initialize(self):
        f = rand_array((self.M, self.N, self.N), self.T)
        kinv = np.array(_poisson_inv_laplacian(self.T, self.N)).reshape(
            1, self.N, self.N
        )
        return (f, kinv)

    def run(self, state):
        f, kinv = state
        uhat = np.fft.fftn(f, axes=(1, 2))
        uhat *= kinv
        return np.fft.ifftn(uhat, axes=(1, 2))


register_benchmark("poisson_fft", PoissonFFT)
