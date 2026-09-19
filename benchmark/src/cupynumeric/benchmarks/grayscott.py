# cuPyNumeric implementation.
import cupynumeric as np
import numpy as host_np

from core import register_benchmark, rand_array, zeros_array, ones_array


class GrayScott:
    name = "grayscott"
    fence_each_iteration = False  # Timesteps belong to one trajectory.

    # dt = dx/5; c_u, c_v, f, k as in grayscott.jl's GSParams defaults.
    def __init__(self, T, N, M, dx=1.0, c_u=1.0, c_v=0.3, f=0.03, k=0.06):
        self.T, self.N, self.M = T, N, M
        self.dx = T(dx)
        self.dt = T(dx / 5)
        self.c_u, self.c_v, self.f, self.k = T(c_u), T(c_v), T(f), T(k)

    def dims(self):
        return self.N, self.M

    def initialize(self):
        u = ones_array((self.N, self.M), self.T)
        v = zeros_array((self.N, self.M), self.T)
        u_new = zeros_array((self.N, self.M), self.T)
        v_new = zeros_array((self.N, self.M), self.T)

        seed = min(150, self.N, self.M)
        u[:seed, :seed] = rand_array((seed, seed), self.T)
        v[:seed, :seed] = rand_array((seed, seed), self.T)
        # mutable list so run() can swap buffers in place
        return [u, v, u_new, v_new]

    def run(self, state):
        u, v, u_new, v_new = state
        # Copy: slice views alias the ping-pong buffers under deferred execution.
        u = u.copy()
        v = v.copy()
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

    def correctness_dims(self):
        n = min(32, self.N, self.M)
        return n, n

    def check_correctness(self):
        # run() is array-generic; drive both backends from one host IC.
        n, _ = self.correctness_dims()
        steps = getattr(self, "n_correctness_iter", 5)
        # Smooth deterministic IC; an iid-random grid is FP-sensitive here.
        seed = min(150, n)
        i = host_np.arange(1, seed + 1, dtype=self.T)
        I, J = host_np.meshgrid(i, i, indexing="ij")
        u0 = host_np.ones((n, n), dtype=self.T)
        v0 = host_np.zeros((n, n), dtype=self.T)
        u0[:seed, :seed] = (0.5 + 0.5 * host_np.sin(I) * host_np.cos(J)).astype(self.T)
        v0[:seed, :seed] = (0.25 + 0.25 * host_np.cos(I) * host_np.sin(J)).astype(self.T)

        def evolve(xp, u, v):
            state = [u, v, xp.zeros((n, n), dtype=self.T), xp.zeros((n, n), dtype=self.T)]
            for _ in range(steps):
                self.run(state)
            return state[0], state[1]

        au, av = evolve(np, np.asarray(u0), np.asarray(v0))
        eu, ev = evolve(host_np, u0.copy(), v0.copy())
        tol = 1e-3 if self.T is np.float32 else 1e-10
        ok = host_np.allclose(host_np.asarray(au), eu, rtol=tol, atol=tol) and \
            host_np.allclose(host_np.asarray(av), ev, rtol=tol, atol=tol)
        return "pass" if ok else "fail"


register_benchmark("grayscott", GrayScott)
