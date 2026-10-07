"""cuPyNumeric equivalent of examples/gray-scott.jl."""

import cupynumeric as np


def step(u, v, dx, dt, c_u, c_v, feed, kill):
    u_new = np.zeros_like(u)
    v_new = np.zeros_like(v)

    u_mid = u[1:-1, 1:-1]
    v_mid = v[1:-1, 1:-1]
    reaction = u_mid * v_mid**2
    f_u = -reaction + feed * (1 - u_mid)
    f_v = reaction - (feed + kill) * v_mid

    u_lap = (
        u[2:, 1:-1] - 2 * u_mid + u[:-2, 1:-1]
        + u[1:-1, 2:] - 2 * u_mid + u[1:-1, :-2]
    ) / dx**2
    v_lap = (
        v[2:, 1:-1] - 2 * v_mid + v[:-2, 1:-1]
        + v[1:-1, 2:] - 2 * v_mid + v[1:-1, :-2]
    ) / dx**2

    u_new[1:-1, 1:-1] = (c_u * u_lap + f_u) * dt + u_mid
    v_new[1:-1, 1:-1] = (c_v * v_lap + f_v) * dt + v_mid

    u_new[:, 0] = u[:, -2]
    u_new[:, -1] = u[:, 1]
    u_new[0, :] = u[-2, :]
    u_new[-1, :] = u[1, :]
    v_new[:, 0] = v[:, -2]
    v_new[:, -1] = v[:, 1]
    v_new[0, :] = v[-2, :]
    v_new[-1, :] = v[1, :]
    return u_new, v_new


def gray_scott(n=4000, n_steps=100):
    dx = 1.0
    dt = dx / 5
    u = np.ones((n, n))
    v = np.zeros((n, n))
    seed = min(150, n)
    u[:seed, :seed] = np.random.rand(seed, seed)
    v[:seed, :seed] = np.random.rand(seed, seed)

    for _ in range(n_steps):
        u, v = step(u, v, dx, dt, 1.0, 0.3, 0.03, 0.06)
    return u, v


if __name__ == "__main__":
    gray_scott()
