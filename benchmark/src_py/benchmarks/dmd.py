import cupynumeric as np

from core import register_benchmark, rand_array


class DMD:
    name = "dmd_baseline"

    def __init__(self, T, N, M):
        self.T, self.N, self.M = T, N, M
        self.r = min(20, M - 1)

    def dims(self):
        return self.N, self.M

    def total_flops(self):
        # Same breakdown as benchmark/src/benchmarks/dmd.jl
        m = self.N
        n = self.M - 1
        r = self.r
        return (
            2 * m * n * n
            + 11 * n * n * n
            + 2 * m * n * r
            + m * r
            + 2 * m * r * r
            + 25 * r * r * r
            + 2 * m * r * r
        )

    def initialize(self):
        X = rand_array((self.N, self.M), self.T)
        return (X,)

    def run(self, state):
        (X,) = state
        n = self.M - 1
        X1 = X[:, :n]
        X2 = X[:, 1:]
        U, S, Vt = np.linalg.svd(X1, full_matrices=False)
        r = self.r
        U = U[:, :r]
        Vt = Vt[:r, :]
        B = (X2 @ Vt.T) * (self.T(1) / S[:r])
        At = U.T @ B
        _, W = np.linalg.eig(At)
        ctype = np.complex64 if self.T == np.float32 else np.complex128
        _ = B.astype(ctype) @ W


register_benchmark("dmd_baseline", DMD)
