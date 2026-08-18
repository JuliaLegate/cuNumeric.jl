#= Copyright 2026 Northwestern University,
 *                   Carnegie Mellon University University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author(s): David Krasowska <krasow@u.northwestern.edu>
 *            Ethan Meitz <emeitz@andrew.cmu.edu>
=#

#= Periodic Poisson via FFT.

Solve ∇²u = f on the unit square with periodic boundaries. In Fourier space
that is a pointwise divide: û[k] = f̂[k] / (−|k|²), with the k = 0 mode set
to zero so the potential has mean zero.

A manufactured solution u = sin(2πx) sin(2πy) has ∇²u = −8π² u, so we can
check the residual after one forward FFT, the divide, and one inverse FFT.

The same kernel on a stack of right-hand sides is `batched_fft` — each slice
is an independent electrostatics / gravity solve, and the batch axis is what
can split across GPUs.
=#

using cuNumeric
using Printf

function integer_fftfreq(n::Int)
    n2 = n ÷ 2
    return iseven(n) ? vcat(0:(n2 - 1), (-n2):-1) : vcat(0:n2, (-n2):-1)
end

# Multipliers for û = f̂ / (−|k|²). DC is 0 (mean-zero potential).
function poisson_inv_laplacian(::Type{T}, n::Int) where {T}
    freq = integer_fftfreq(n)
    invk = Matrix{T}(undef, n, n)
    s = T(4 * π^2)
    for j in 1:n, i in 1:n
        k2 = s * T(freq[i]^2 + freq[j]^2)
        invk[i, j] = k2 == 0 ? zero(T) : -inv(k2)
    end
    return invk
end

function poisson_solve(f::NDArray)
    n, m = size(f)
    n == m || throw(ArgumentError("poisson_solve expects a square grid"))
    invk = NDArray(poisson_inv_laplacian(eltype(f), n))
    uhat = fft(f)
    return real(ifft(uhat .* invk))
end

function main()
    n = 64
    xs = range(0, 1; length=n + 1)[1:(end - 1)]
    u_true = Float32[sin(2π * x) * sin(2π * y) for x in xs, y in xs]
    f_h = (-8 * Float32(π)^2) .* u_true

    f = NDArray(f_h)
    u = poisson_solve(f)
    err = maximum(abs.(u - NDArray(u_true)))
    @printf("single grid %dx%d  max |u − u_true| = %.3e\n", n, n, unwrap(err))

    # Four independent charge distributions, one FFT task, batch axis first.
    b = 4
    f_batch = NDArray(repeat(reshape(f_h, 1, n, n), b, 1, 1))
    invk = reshape(NDArray(poisson_inv_laplacian(Float32, n)), 1, n, n)
    u_batch = real(cuNumeric.batched_ifft(cuNumeric.batched_fft(f_batch) .* invk))
    err_b = maximum(abs.(u_batch - NDArray(repeat(reshape(u_true, 1, n, n), b, 1, 1))))
    @printf("batched  %d×%dx%d  max |u − u_true| = %.3e\n", b, n, n, unwrap(err_b))

    return u
end

main()
