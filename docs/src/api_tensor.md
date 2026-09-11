# Tensor Contractions

We extend [TensorOperations.jl](https://quantumkithub.github.io/TensorOperations.jl/stable/) to work with `NDArray`s. Loading TensorOperations alongside cuNumeric activates the package extension.

Useful macros if you are new to TensorOperations are
[`@tensor`](https://quantumkithub.github.io/TensorOperations.jl/stable/man/indexnotation/#The-@tensor-macro),
[`@tensoropt`](https://quantumkithub.github.io/TensorOperations.jl/stable/man/indexnotation/#TensorOperations.@tensoropt),
and
[`@notensor`](https://quantumkithub.github.io/TensorOperations.jl/stable/man/indexnotation/#TensorOperations.@notensor).

The extension supports additions and permutations, traces, pairwise
contractions, complex conjugation, scale factors, accumulation into an existing
tensor, and multi-step `@tensor` blocks. Prefer `@tensor` (with `opt=true` for
larger networks) over the low-level `contract` / `contract!` API at the bottom
of this page.

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

Prefer a Julia `Number` for scale factors when possible (i.e., `2.0f0`, `randn()`, …). This allows TensorOperations.jl to perform optimizatoins for special values like zero and one.

A 0D `NDArray` is also accepted as a scale — for example `sum(C)`, or a
fully contracted `@tensor` result. If you intend to use these results in down-stream tensor contractions keep those on device. `unwrap` copies the value to the host and **blocks the Legate runtime**, so only unwrap
when you truly need a Julia `Number` (i.e. printing of host-side if-else).

```julia
A = cuNumeric.rand(Float64, 64, 32)
B = cuNumeric.rand(Float64, 32, 16)

@tensor opt=true C[i, j] := A[i, k] * B[k, j]   # NDArray

α = 2.0
@tensor D[i, j] := α * C[i, j]          # Julia Number, no sync

@tensor s = conj(C[i, j]) * C[i, j]    # 0D NDArray, stays asynchronous
@tensor E[i, j] := s * C[i, j]         # reuse 0D as a scale, still no unwrap
x = unwrap(s)                          # host Number; blocks
```

## Low-level `contract` / `contract!`

> [!WARNING]
> Prefer `@tensor`. `contract` and `contract!` are the pairwise primitive the
> extension calls. They do not parse einsum strings, pick a contraction order,
> or free intermediates for you.

Mode labels are not einsum strings: `"ik"` with `"kj"` is a GEMM; `"ijk"` with
`"ikl"` and explicit output `"ijl"` is a batched product. Labels may be ASCII
strings, `Char` tuples, or integers (`1` maps to `'a'`).

```julia
A = cuNumeric.rand(Float32, 64, 32)
B = cuNumeric.rand(Float32, 32, 16)
C = contract(A, "ik", B, "kj")              # allocates, Einstein output order
contract!(similar(C), "ij", A, "ik", B, "kj"; α=2, β=0)

# batched: keep the shared 'i' on the output
AA = cuNumeric.rand(Float32, 8, 16, 32)
BB = cuNumeric.rand(Float32, 8, 32, 4)
CC = cuNumeric.zeros(Float32, 8, 16, 4)
contract!(CC, "ijl", AA, "ijk", BB, "ikl")

tensordot(AA, BB, ([3], [2]))               # same contraction, axes form
```

`α` and `β` implement `C = β*C + α*(A ⋆ B)` in Julia. Prefer a Julia `Number`;
a 0-d `NDArray` is also accepted. The C++ kernel always writes the unscaled
product; `β ≠ 0` uses a temporary. Duplicate labels inside one array are not
allowed — use `cuNumeric.diagonal` first.

This path is multi-GPU via Legate tiling (per-tile cuTENSOR or TBLIS), not
cuTensorMp. Integer and `Bool` inputs promote to `Float64` like other linalg.

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/contract.jl"]
```
