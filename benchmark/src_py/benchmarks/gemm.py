import cupynumeric as np

from core import register_benchmark


class GEMM:
    name = "gemm"

    def __init__(self, T, N, M):
        self.T, self.N, self.M = T, N, M

    def dims(self):
        return self.N, self.M

    def total_flops(self):
        return self.N * self.N * (2 * self.M - 1)

    def initialize(self):
        A = np.random.rand(self.N, self.M).astype(self.T)
        B = np.random.rand(self.M, self.N).astype(self.T)
        C = np.zeros((self.N, self.N), dtype=self.T)
        return (C, A, B)

    def run(self, state):
        C, A, B = state
        np.matmul(A, B, out=C)


register_benchmark("gemm", GEMM)
