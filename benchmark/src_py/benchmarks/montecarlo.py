import cupynumeric as np

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


register_benchmark("montecarlo", MonteCarlo)
