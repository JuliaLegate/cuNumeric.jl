# cuPyNumeric implementation.
import cupynumeric as np
import numpy as host_np
import math

from core import register_benchmark, rand_array


class MonteCarlo:
    name = "montecarlo"

    def __init__(self, T, N, M):
        self.T = T
        self.n_samples = N

    def dims(self):
        return self.n_samples, 1

    def initialize(self):
        x = (self.T(10) * rand_array(self.n_samples, self.T))
        return (x,)

    def run(self, state):
        (x,) = state
        return (self.T(10) / self.n_samples) * np.sum(np.exp(-(x * x)))

    def check_correctness(self):
        n = min(self.n_samples, 1024)
        host_samples = host_np.linspace(0, 10, n, dtype=self.T)
        actual = float(
            (self.T(10) / n)
            * np.sum(np.exp(-(np.asarray(host_samples) * np.asarray(host_samples))))
        )
        expected = float(
            (self.T(10) / n)
            * sum(math.exp(-(float(x) * float(x))) for x in host_samples)
        )
        tolerance = 1e-3 if self.T is np.float32 else 1e-10
        return "pass" if math.isclose(
            actual, expected, rel_tol=tolerance, abs_tol=tolerance
        ) else "fail"


register_benchmark("montecarlo", MonteCarlo)
