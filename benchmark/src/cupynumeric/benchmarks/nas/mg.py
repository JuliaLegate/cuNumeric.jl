"""NAS MG using cuPyNumeric views and array expressions.

LIMITATION: The exact NPB sparse right-hand side is generated on the host, as
in NPB-GPU, and uploaded before timing. All residual, restriction,
interpolation, smoothing, and periodic-boundary work stays in cuPyNumeric.
Transfers use separable axis passes and temporary arrays, not JACC's direct
per-cell kernels; array expressions also materialize intermediates. The common
harness times initial zeroing and L2 sum-of-squares, but omits NPB's Linf norm;
see nas/README.md.
"""

import math

import cupynumeric as np
import numpy as host_np

from core import register_benchmark


SEED = 314159265.0
MULTIPLIER = 1220703125.0
EXTREMA = 10
A = (-8.0 / 3.0, 0.0, 1.0 / 6.0, 1.0 / 12.0)
CLASSES = {
    "S": (32, 4, 0.5307707005734e-4),
    "W": (128, 4, 0.6467329375339e-5),
    "A": (256, 4, 0.2433365309069e-5),
    "B": (256, 20, 0.1800564401355e-5),
    "C": (512, 20, 0.5706732285740e-6),
    "D": (1024, 50, 0.1583275060440e-9),
    "E": (2048, 50, 0.8157592357404e-10),
}


def randlc(x, a=MULTIPLIER):
    r23, t23, r46, t46 = 2.0**-23, 2.0**23, 2.0**-46, 2.0**46
    a1 = math.trunc(r23 * a)
    a2 = a - t23 * a1
    x1 = math.trunc(r23 * x)
    x2 = x - t23 * x1
    t1 = a1 * x2 + a2 * x1
    z = t1 - t23 * math.trunc(r23 * t1)
    t3 = t23 * z + a2 * x2
    nxt = t3 - t46 * math.trunc(r46 * t3)
    return nxt, r46 * nxt


def insert_extreme(values, indices, value, index, largest):
    if (largest and value <= values[0]) or (not largest and value >= values[0]):
        return
    values[0], indices[0] = value, index
    for i in range(len(values) - 1):
        ordered = values[i] <= values[i + 1] if largest else values[i] >= values[i + 1]
        if ordered:
            break
        values[i], values[i + 1] = values[i + 1], values[i]
        indices[i], indices[i + 1] = indices[i + 1], indices[i]


def rhs_host(n):
    lows, highs = [1.0] * EXTREMA, [0.0] * EXTREMA
    low_indices, high_indices = [(0, 0, 0)] * EXTREMA, [(0, 0, 0)] * EXTREMA
    seed = SEED
    for k in range(1, n + 1):
        for j in range(1, n + 1):
            for i in range(1, n + 1):
                seed, value = randlc(seed)
                index = (i, j, k)
                insert_extreme(lows, low_indices, value, index, False)
                insert_extreme(highs, high_indices, value, index, True)
    rhs = host_np.zeros((n + 2, n + 2, n + 2), dtype=host_np.float64)
    for index in low_indices:
        rhs[index] = -1.0
    for index in high_indices:
        rhs[index] = 1.0
    comm3(rhs)
    return rhs


def comm3(u):
    u[0, 1:-1, 1:-1] = u[-2, 1:-1, 1:-1]
    u[-1, 1:-1, 1:-1] = u[1, 1:-1, 1:-1]
    u[:, 0, 1:-1] = u[:, -2, 1:-1]
    u[:, -1, 1:-1] = u[:, 1, 1:-1]
    u[:, :, 0] = u[:, :, -2]
    u[:, :, -1] = u[:, :, 1]
    return u


def resid(r, u, v):
    c = (slice(1, -1),) * 3
    xm, xp = slice(0, -2), slice(2, None)
    m, p = slice(0, -2), slice(2, None)
    mid = slice(1, -1)
    r[c] = (
        v[c]
        - A[0] * u[c]
        - A[2]
        * (
            u[mid, m, m]
            + u[mid, p, m]
            + u[mid, m, p]
            + u[mid, p, p]
            + u[xm, mid, m]
            + u[xp, mid, m]
            + u[xm, mid, p]
            + u[xp, mid, p]
            + u[xm, m, mid]
            + u[xp, m, mid]
            + u[xm, p, mid]
            + u[xp, p, mid]
        )
        - A[3]
        * (
            u[xm, m, m]
            + u[xp, m, m]
            + u[xm, p, m]
            + u[xp, p, m]
            + u[xm, m, p]
            + u[xp, m, p]
            + u[xm, p, p]
            + u[xp, p, p]
        )
    )
    return comm3(r)


def psinv(u, r, coeff):
    c = (slice(1, -1),) * 3
    xm, xp, mid = slice(0, -2), slice(2, None), slice(1, -1)
    m, p = slice(0, -2), slice(2, None)
    u[c] += (
        coeff[0] * r[c]
        + coeff[1]
        * (
            r[xm, mid, mid]
            + r[xp, mid, mid]
            + r[mid, m, mid]
            + r[mid, p, mid]
            + r[mid, mid, m]
            + r[mid, mid, p]
        )
        + coeff[2]
        * (
            r[mid, m, m]
            + r[mid, p, m]
            + r[mid, m, p]
            + r[mid, p, p]
            + r[xm, mid, m]
            + r[xp, mid, m]
            + r[xm, mid, p]
            + r[xp, mid, p]
            + r[xm, m, mid]
            + r[xp, m, mid]
            + r[xm, p, mid]
            + r[xp, p, mid]
        )
    )
    return comm3(u)


def axis_last(array, axis):
    if axis == 2:
        return array, (0, 1, 2)
    permutation = (1, 2, 0) if axis == 0 else (0, 2, 1)
    inverse = tuple(permutation.index(i) for i in range(3))
    return np.transpose(array, permutation), inverse


def restrict_axis(array, axis):
    back, inverse = axis_last(array, axis)
    d1, d2, n = back.shape
    physical = n - 2
    # Legate cannot reshape a sliced store. Materialize two contiguous slabs,
    # pair neighboring cells in the last dimension, then reduce each pair.
    left = back[:, :, 1 : n - 1].copy()
    right = back[:, :, 2:n].copy()
    paired_shape = (d1, d2, physical // 2, 2)
    reduced = left.reshape(paired_shape).sum(axis=3) + right.reshape(paired_shape).sum(
        axis=3
    )
    return reduced if axis == 2 else np.transpose(reduced, inverse)


def restrict(coarse, fine):
    reduced = restrict_axis(fine, 0)
    reduced = restrict_axis(reduced, 1)
    reduced = restrict_axis(reduced, 2)
    coarse[1:-1, 1:-1, 1:-1] = reduced / 16.0
    return comm3(coarse)


def interp_axis(array, axis, weights):
    back, inverse = axis_last(array, axis)
    d1, d2, n = back.shape
    lo = back[:, :, : n - 1].copy().reshape(d1, d2, n - 1, 1)
    hi = back[:, :, 1:n].copy().reshape(d1, d2, n - 1, 1)
    interpolated = (lo + weights * (hi - lo)).reshape(d1, d2, 2 * (n - 1))
    result = interpolated if axis == 2 else np.transpose(interpolated, inverse)
    return result.copy()


def interp(fine, coarse, weights):
    interpolated = interp_axis(coarse, 0, weights)
    interpolated = interp_axis(interpolated, 1, weights)
    interpolated = interp_axis(interpolated, 2, weights)
    fine += interpolated


def norm2(residual):
    interior = residual[1:-1, 1:-1, 1:-1]
    return np.sum(interior * interior)


class NASMultiGrid:
    name = "nas_mg"

    def __init__(self, T, N, M, **kwargs):
        self.T, self.N, self.M = T, N, M
        self.class_name = str(kwargs.pop("class", "S")).upper()
        if kwargs:
            raise ValueError(f"Unknown NAS MG options: {', '.join(kwargs)}")
        if self.class_name not in CLASSES:
            raise ValueError(f"Unknown NAS MG class {self.class_name}")
        n = CLASSES[self.class_name][0]
        if T is not np.float64 or (N, M) != (n, n):
            raise ValueError(
                f"NAS MG class {self.class_name} requires Float64, N=M={n}"
            )

    def dims(self):
        return self.N, self.M

    def initialize(self):
        n = CLASSES[self.class_name][0]
        sizes = [2**level + 2 for level in range(1, int(math.log2(n)) + 1)]
        return {
            "u": [np.zeros((s, s, s), dtype=np.float64) for s in sizes],
            "r": [np.zeros((s, s, s), dtype=np.float64) for s in sizes],
            "rhs": np.asarray(rhs_host(n)),
            "weights": np.asarray([0.0, 0.5]).reshape(1, 1, 1, 2),
        }

    def run(self, state):
        niter = CLASSES[self.class_name][1]
        coeff = (
            (-3.0 / 8.0, 1.0 / 32.0, -1.0 / 64.0, 0.0)
            if self.class_name in ("S", "W", "A")
            else (-3.0 / 17.0, 1.0 / 33.0, -1.0 / 61.0, 0.0)
        )
        u, r, rhs, weights = state["u"], state["r"], state["rhs"], state["weights"]
        for level in u:
            level.fill(0.0)
        resid(r[-1], u[-1], rhs)
        norm2(r[-1])
        for _ in range(niter):
            for level in range(len(u) - 1, 0, -1):
                restrict(r[level - 1], r[level])
            u[0].fill(0.0)
            psinv(u[0], r[0], coeff)
            for level in range(1, len(u) - 1):
                u[level].fill(0.0)
                interp(u[level], u[level - 1], weights)
                resid(r[level], u[level], r[level])
                psinv(u[level], r[level], coeff)
            interp(u[-1], u[-2], weights)
            resid(r[-1], u[-1], rhs)
            psinv(u[-1], r[-1], coeff)
            resid(r[-1], u[-1], rhs)
        return norm2(r[-1])

    def correctness_dims(self):
        return self.N, self.M

    def check_correctness(self):
        n, _, reference = CLASSES[self.class_name]
        squared = self.run(self.initialize())
        norm = math.sqrt(float(host_np.asarray(squared)) / n**3)
        return "pass" if abs((norm - reference) / reference) <= 1.0e-8 else "fail"


register_benchmark("nas_mg", NASMultiGrid)
