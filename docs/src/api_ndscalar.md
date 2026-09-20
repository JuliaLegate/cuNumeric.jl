# Device scalars and autounwrapping

Full reductions such as `sum(A)`, `dot(x, y)`, and `norm(x)` return an
`NDScalar`. `NDReal{T} <: Real` holds real results, including Boolean results;
`NDComplex{T} <: Number` holds complex results. `NDScalar` is a union of these
two wrapper types. Each owns a reference to a 0D NDArray, with no copy or host
extraction when it is wrapped. Reductions that retain dimensions still return
NDArrays. Explicit 0D array construction and array broadcasts remain arrays.

```julia
using cuNumeric, LinearAlgebra
x = cuNumeric.NDArray([1.0, 2.0, 3.0])
s = sum(x)                    # NDReal{Float64}, accepted in <:Real fields
t = s^2 / 2                   # another device scalar
y = x .* t                    # backend broadcast; no implicit host extraction
value = unwrap(t)             # explicit synchronization, always permitted
```

`ndscalar(a)` wraps an existing 0D NDArray; `s.value` accesses its backend array
without synchronizing. Use `unwrap(s)` or `only(s)` for explicit host extraction.
Arithmetic, `min`/`max`, and promotion between numeric types keep results on the
backend. Showing a wrapper prints its type and a device-scalar marker, not its
value. The existing promotion policy still applies.

Value-dependent conversions, such as floating point to integer or complex to
real, require autounwrap permission even when the target is another NDScalar.
They preserve Julia's `InexactError` checks instead of silently truncating values.

## Scoped permission

Implicit host extraction is disabled by default. Comparisons, scalar predicates,
and conversion to supported host numeric types require `autounwrap` permission:

```julia
@autounwrap s > 0              # Julia Bool
autounwrap() do
    Float64(s)                 # Julia Float64
end
```

The scope returns the body's result and restores the previous permission even
when the body throws. Nested `autounwrap(false) do ... end` disables extraction
temporarily. `autounwrap(true)` / `autounwrap(false)` set the calling task's
permission until changed again. Independent tasks do not inherit permission.
This permission is separate from `allowscalar` and `allowpromotion`.

Arithmetic remains asynchronous even inside an autounwrap scope. There is no
general fallback that unwraps arguments when Julia cannot find a method.
For example, a function accepting only `Float64` still needs `f(Float64(s))`.
Julia also requires an actual Bool in `if`: use `Bool(all(A))` within the scope,
or a comparison that returns a host Bool. Autounwrapping does not change Julia's
dispatch or condition evaluation rules.

## IterativeSolvers

The wrappers are intended to satisfy existing `Real` and `Number` constraints
without changing solver source. `dev/ndscalar_cg.jl` tests unmodified
IterativeSolvers 0.9.4 with scalar indexing disabled. With compatible operations,
the intended use is:

```julia
using IterativeSolvers
x = @autounwrap cg(A, b)
```

This is compatibility with a scalar-oriented solver, not a guarantee of one
synchronization per iteration. Its convergence comparisons extract a host Bool;
`log=true` also converts residuals to host history values. Stock PCG initializes
its `ρ` field with a host number, so assigning a device dot product to that
field converts it under the same permission. Complex solves may also need an
`allowpromotion` scope under the package's existing promotion policy.
