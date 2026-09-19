# cuPyNumeric implementation.
import cupynumeric as np
import numpy as host_np

from core import register_benchmark, rand_array, zeros_array


class GEMM:
    name = "gemm"

    def __init__(self, T, N, M):
        self.T, self.N, self.M = T, N, M

    def dims(self):
        return self.N, self.M

    def correctness_dims(self):
        return min(self.N, 8), min(self.M, 8)

    def initialize(self):
        A = rand_array((self.N, self.M), self.T)
        B = rand_array((self.M, self.N), self.T)
        C = zeros_array((self.N, self.N), self.T)
        return (C, A, B)

    def run(self, state):
        C, A, B = state
        np.matmul(A, B, out=C)

    def check_correctness(self):
        n, m = self.correctness_dims()
        A = host_np.arange(1, n * m + 1, dtype=self.T).reshape((n, m), order="F")
        B = host_np.arange(1, m * n + 1, dtype=self.T).reshape((m, n), order="F")
        A /= self.T(n * m)
        B /= self.T(m * n)
        actual = host_np.asarray(np.matmul(np.asarray(A), np.asarray(B)))
        expected = A @ B
        tolerance = 1e-3 if self.T is np.float32 else 1e-10
        return "pass" if host_np.allclose(
            actual, expected, rtol=tolerance, atol=tolerance
        ) else "fail"


register_benchmark("gemm", GEMM)
