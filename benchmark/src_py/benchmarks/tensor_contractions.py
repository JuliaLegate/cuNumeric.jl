import cupynumeric as np

from core import register_benchmark, rand_array, zeros_array


def _optimal_path(expression, *operands):
    path, _ = np.einsum_path(
        expression, *operands, optimize="optimal"
    )
    # cuPyNumeric returns NumPy's sentinel-prefixed path, but its einsum forwards
    # paths directly to opt_einsum, which expects only contraction tuples.
    return path[1:] if path and path[0] == "einsum_path" else path


class TensorProjection3:
    name = "tensor_projection3"
    expression = "ijk,ni,mj,lk->nml"

    def __init__(self, T, N, M):
        self.T, self.N = T, N

    def dims(self):
        return self.N, 1

    def total_flops(self):
        return 3 * self.N**3 * (2 * self.N - 1)

    def initialize(self):
        A = rand_array((self.N, self.N, self.N), self.T)
        B = rand_array((self.N, self.N), self.T)
        D = zeros_array((self.N, self.N, self.N), self.T)
        path = _optimal_path(self.expression, A, B, B, B)
        return D, A, B, path

    def run(self, state):
        D, A, B, path = state
        np.einsum(
            self.expression, A, B, B, B, out=D, optimize=path
        )


class TensorContract4:
    name = "tensor_contract4"
    expression = "aicj,ibjd->abcd"

    def __init__(self, T, N, M):
        self.T, self.N = T, N

    def dims(self):
        return self.N, 1

    def total_flops(self):
        return self.N**4 * (2 * self.N**2 - 1)

    def initialize(self):
        X = rand_array((self.N, self.N, self.N, self.N), self.T)
        Y = rand_array((self.N, self.N, self.N, self.N), self.T)
        C = zeros_array((self.N, self.N, self.N, self.N), self.T)
        path = _optimal_path(self.expression, X, Y)
        return C, X, Y, path

    def run(self, state):
        C, X, Y, path = state
        np.einsum(self.expression, X, Y, out=C, optimize=path)


register_benchmark("tensor_projection3", TensorProjection3)
register_benchmark("tensor_contract4", TensorContract4)
