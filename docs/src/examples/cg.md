# Conjugate gradient

Conjugate gradient solves ``Ax=b`` for a real symmetric positive-definite
matrix using matrix-vector products, reductions and vector updates. The complete recurrence is shown below.

```julia
using cuNumeric, LinearAlgebra

function cg!(x, A, b; rtol=1e-8, check_every=10, max_iter=1000)
    r = b - A*x
    p, Ap = copy(r), similar(r)
    rho = sum(r .* r)
    target = rtol^2 * only(sum(b .* b))
    only(rho) <= target && return x

    for k in 1:max_iter
        mul!(Ap, A, p)
        # Protect zero denominators if convergence occurs between checks.
        alpha = rho ./ max.(sum(p .* Ap), floatmin(eltype(x)))
        x .+= alpha .* p
        r .-= alpha .* Ap
        next = sum(r .* r)
        beta = next ./ max.(rho, floatmin(eltype(x)))
        p .= r .+ beta .* p
        rho = next

        if k % check_every == 0 || k == max_iter
            only(rho) <= target && return x
        end
    end
    error("CG did not converge within max_iter")
end

A = NDArray([4.0 1.0 1.0; 1.0 3.0 0.5; 1.0 0.5 2.0])
# cuNumeric's matrix multiplication uses single-column matrices for vectors.
b = cuNumeric.ones(Float64, 3, 1)
x = cuNumeric.zeros(Float64, 3, 1)
cg!(x, A, b; check_every=5, max_iter=100)

# Compare the iterative result with the direct solve API.
x_direct = cuNumeric.solve(A, b)
println(Array(x))
@assert isapprox(Array(x), Array(x_direct); rtol=1e-8)
```

`rho` holds the squared residual norm. Reductions and the coefficients `alpha`
and `beta` remain in cuNumeric's computation graph; `only(rho)` reads a value
back to the host at a convergence check. Increasing `check_every` lets the host
submit more iterations ahead, at the cost of potentially doing extra work
before observing convergence. A check also occurs at the iteration limit.
