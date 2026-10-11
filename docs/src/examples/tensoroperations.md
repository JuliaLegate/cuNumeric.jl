# TensorOperations.jl

We extend [TensorOperations.jl](https://quantumkithub.github.io/TensorOperations.jl/stable/) to work with `NDArray`s. Loading TensorOperations alongside cuNumeric activates the package extension.

Useful macros if you are new to TensorOperations are
[`@tensor`](https://quantumkithub.github.io/TensorOperations.jl/stable/man/indexnotation/#The-@tensor-macro),
[`@tensoropt`](https://quantumkithub.github.io/TensorOperations.jl/stable/man/indexnotation/#TensorOperations.@tensoropt),
and
[`@notensor`](https://quantumkithub.github.io/TensorOperations.jl/stable/man/indexnotation/#TensorOperations.@notensor).

The extension supports additions and permutations, traces, pairwise
contractions, complex conjugation, scale factors, accumulation into an existing
tensor, and multi-step `@tensor` blocks. Prefer `@tensor` (with `opt=true` for
larger networks) over the [low-level contraction API](../api_tensor.md).

```julia
using cuNumeric
using TensorOperations

α = randn() # prefer a Julia Number when you already have one
A = cuNumeric.randn(5, 5, 5, 5, 5, 5)
B = cuNumeric.randn(5, 5, 5)
C = cuNumeric.randn(5, 5, 5)
D = cuNumeric.zeros(5, 5, 5)

@tensor begin
    D[a, b, c] = A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
    E[a, b, c] := A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
end
```

## Scalars

Prefer a Julia `Number` for scale factors when possible (i.e., `2.0f0`, `randn()`, …). This allows TensorOperations.jl to perform optimizations for special values like zero and one.

A `CNScalar`, such as `sum(C)`, or a raw 0D `NDArray`, such as a
fully contracted `@tensor` result, is also accepted as a scale. Keep these results in runtime-managed storage when reusing them in tensor contractions. `fetch` retrieves a native Julia scalar and waits for the result, so only fetch
when you truly need a Julia `Number` (e.g. printing or a host-side conditional).

```julia
A = cuNumeric.rand(Float64, 64, 32)
B = cuNumeric.rand(Float64, 32, 16)

@tensor opt=true C[i, j] := A[i, k] * B[k, j]   # NDArray

α = 2.0
@tensor D[i, j] := α * C[i, j]          # Julia Number, no sync

@tensor s = conj(C[i, j]) * C[i, j]    # 0D NDArray, stays asynchronous
@tensor E[i, j] := s * C[i, j]         # reuse 0D as a scale, still no fetch
x = fetch(s)                          # host Number; blocks
```
