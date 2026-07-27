import cupynumeric as np

from core import register_benchmark


class GrayScott:
    name = "grayscott"

    # dt = dx/5; c_u, c_v, f, k as in grayscott.jl's GSParams defaults.
    def __init__(self, T, N, M, dx=1.0, c_u=1.0, c_v=0.3, f=0.03, k=0.06):
        self.T, self.N, self.M = T, N, M
        self.dx = T(dx)
        self.dt = T(dx / 5)
        self.c_u, self.c_v, self.f, self.k = T(c_u), T(c_v), T(f), T(k)

    def dims(self):
        return self.N, self.M

    def total_flops(self):
        return self.N * self.M

    def initialize(self):
        d = (self.N, self.M)
        u = np.ones(d, dtype=self.T)
        v = np.zeros(d, dtype=self.T)
        u_new = np.zeros(d, dtype=self.T)
        v_new = np.zeros(d, dtype=self.T)

        seed = min(150, self.N, self.M)
        u[:seed, :seed] = np.random.rand(seed, seed).astype(self.T)
        v[:seed, :seed] = np.random.rand(seed, seed).astype(self.T)
        # mutable list so run() can swap buffers in place
        return [u, v, u_new, v_new]

    def run(self, state):
        u, v, u_new, v_new = state
        ui = u[1:-1, 1:-1]
        vi = v[1:-1, 1:-1]

        F_u = (-ui * (vi * vi)) + self.f * (1 - ui)
        F_v = (ui * (vi * vi)) - (self.f + self.k) * vi

        dx2 = self.dx * self.dx
        u_lap = (
            (u[2:, 1:-1] - 2 * ui + u[:-2, 1:-1]) / dx2
            + (u[1:-1, 2:] - 2 * ui + u[1:-1, :-2]) / dx2
        )
        v_lap = (
            (v[2:, 1:-1] - 2 * vi + v[:-2, 1:-1]) / dx2
            + (v[1:-1, 2:] - 2 * vi + v[1:-1, :-2]) / dx2
        )

        u_new[1:-1, 1:-1] = (self.c_u * u_lap + F_u) * self.dt + ui
        v_new[1:-1, 1:-1] = (self.c_v * v_lap + F_v) * self.dt + vi

        # periodic boundary conditions
        u_new[:, 0] = u[:, -2]
        u_new[:, -1] = u[:, 1]
        u_new[0, :] = u[-2, :]
        u_new[-1, :] = u[1, :]
        v_new[:, 0] = v[:, -2]
        v_new[:, -1] = v[:, 1]
        v_new[0, :] = v[-2, :]
        v_new[-1, :] = v[1, :]

        # swap references rather than copy
        state[0], state[2] = u_new, u
        state[1], state[3] = v_new, v


register_benchmark("grayscott_baseline", GrayScott)
