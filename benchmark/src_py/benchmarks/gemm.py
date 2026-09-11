import cupynumeric as np

from core import register_benchmark, rand_array, zeros_array


class GEMM:
    name = "gemm"

    def __init__(self, T, N, M):
        self.T, self.N, self.M = T, N, M

    def dims(self):
        return self.N, self.M

    def initialize(self):
        A = rand_array((self.N, self.M), self.T)
        B = rand_array((self.M, self.N), self.T)
        C = zeros_array((self.N, self.N), self.T)
        return (C, A, B)

    def run(self, state):
        C, A, B = state
        np.matmul(A, B, out=C)


register_benchmark("gemm", GEMM)
