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

α = randn() # Julia scalar, not a 0D NDArray
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

`@tensor` does **not** use the same 0D-`NDArray` convention as the rest of cuNumeric. In the future we will break the TensorOperatoins API so that scalars are 0D Arrays instead.

```@raw html
<ol>
<li><strong>Scale factors must be Julia scalars.</strong> Coefficients such as <code>α</code> in <code>α * C[i, j]</code> must be a Julia <code>Number</code>. A 0D <code>NDArray</code> (for example the result of <code>sum</code>) is not accepted.</li>
<li><strong>A fully contracted result is a Julia scalar.</strong> When every index is contracted, <code>@tensor</code> returns a host <code>Number</code>, not a 0D <code>NDArray</code>. TensorOperations gets that value with <code>tensorscalar</code>, which indexes the rank-zero array and therefore needs <code>@allowscalar</code>.</li>
</ol>
```

```julia
A = cuNumeric.rand(Float64, 64, 32)
B = cuNumeric.rand(Float64, 32, 16)

@tensor opt=true C[i, j] := A[i, k] * B[k, j]   # NDArray

α = 2.0                                         # OK
# α = sum(C)                                    # 0D NDArray, not a scale factor
@tensor D[i, j] := α * C[i, j]

@allowscalar @tensor s = conj(C[i, j]) * C[i, j]  # Julia Number, blocks
```

Full contractions also force a host synchronization, which stalls the Legate
runtime. If you can keep a 0D `NDArray` instead, use an ordinary reduction:

```julia
s = sum(conj(C) .* C)   # NDArray{T,0}, stays asynchronous
x = unwrap(s)           # host Number, only when you need it
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

`α` and `β` implement `C = β*C + α*(A ⋆ B)` in Julia. The C++ kernel always
writes the unscaled product; `β ≠ 0` uses a temporary. Duplicate labels inside
one array are not allowed — use `cuNumeric.diagonal` first.

This path is multi-GPU via Legate tiling (per-tile cuTENSOR or TBLIS), not
cuTensorMp. Integer and `Bool` inputs promote to `Float64` like other linalg.

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/contract.jl"]
```
