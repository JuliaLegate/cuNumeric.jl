# cuPyNumeric implementation of the tridiag(1,4,1) conjugate-gradient solve.
# Mirrors the array-generic recurrence in ../../benchmarks/cg.jl.
import cupynumeric as np
import numpy as host_np

from core import register_benchmark


class ConjugateGradient:
    name = "cg"

    def __init__(self, T, N, M, check_every=10, max_iter=1000):
        self.T, self.N, self.M = T, N, M
        self.check_every = check_every
        self.max_iter = max_iter

    def dims(self):
        return self.N, 1

    def initialize(self):
        T, N = self.T, self.N
        lower = np.ones(N, dtype=T)
        diagonal = np.full(N, T(4), dtype=T)
        upper = np.ones(N, dtype=T)
        x = np.zeros(N, dtype=T)
        r = np.zeros(N, dtype=T)
        p = np.zeros(N, dtype=T)
        Ap = np.zeros(N, dtype=T)
        return {"A": (lower, diagonal, upper), "x": x, "work": (r, p, Ap)}

    # Solve tridiag(1,4,1)*x = 1/2 from zero; reductions stay deferred until checked.
    def run(self, state):
        T, N = self.T, self.N
        lower, diagonal, upper = state["A"]
        x = state["x"]
        r, p, Ap = state["work"]
        tiny = self.T(host_np.finfo(self.T).tiny)
        x[:] = T(0)
        r[:] = T(0.5)
        p[:] = r
        rho = np.sum(r * r)
        target = (1e-5 if self.T is np.float32 else 1e-8) ** 2 * N / 4
        for k in range(1, self.max_iter + 1):
            Ap[:] = diagonal * p
            Ap[1:] += lower[1:] * p[:-1]
            Ap[:-1] += upper[:-1] * p[1:]
            alpha = rho / np.maximum(np.sum(p * Ap), tiny)
            x += alpha * p
            r -= alpha * Ap
            nxt = np.sum(r * r)
            p[:] = r + (nxt / np.maximum(rho, tiny)) * p
            rho = nxt
            if k % self.check_every == 0 or k == self.max_iter:
                rr = float(rho)
                if not host_np.isfinite(rr):
                    raise RuntimeError("CG produced a nonfinite residual")
                if rr <= target or self.max_iter == 1:
                    return k
        raise RuntimeError("CG did not converge within max_iter")

    def correctness_dims(self):
        n = min(self.N, 32)
        return n, 1

    def check_correctness(self):
        n = min(self.N, 32)
        small = ConjugateGradient(self.T, n, 1, self.check_every, self.max_iter)
        state = small.initialize()
        small.run(state)
        x = host_np.asarray(state["x"])
        A = (
            host_np.diag(host_np.full(n, 4, dtype=self.T))
            + host_np.diag(host_np.ones(n - 1, dtype=self.T), 1)
            + host_np.diag(host_np.ones(n - 1, dtype=self.T), -1)
        )
        if self.max_iter == 1:
            err = x - self.T(n / (12 * n - 4))
        else:
            err = A @ x - self.T(0.5)
        tol = (2e-5 if self.T is np.float32 else 2e-8) * host_np.sqrt(n) / 2
        return "pass" if host_np.linalg.norm(err) <= tol else "fail"


register_benchmark("cg", ConjugateGradient)
