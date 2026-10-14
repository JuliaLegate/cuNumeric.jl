# Tensor Contractions

For Einstein-index notation, supported operations, and scalar handling, see the
[TensorOperations.jl guide](examples/tensoroperations.md).

## Low-level `contract` / `contract!`

> [!WARNING]
> Prefer `@tensor`. `contract` and `contract!` are the pairwise primitive the
> extension calls. They do not parse einsum strings, pick a contraction order,
> or free intermediates for you.

Mode labels are not einsum strings: `"ik"` with `"kj"` is a GEMM. `"ijk"` with
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

`α` and `β` implement `C = β*C + α*(A ⋆ B)` in Julia. Prefer a Julia `Number`.
A 0D `NDArray` is also accepted. The C++ kernel always writes the unscaled
product. `β ≠ 0` uses a temporary. Duplicate labels inside one array are not
allowed. Use `cuNumeric.diagonal` first.

This path is multi-GPU via Legate tiling (per-tile cuTENSOR or TBLIS), not
cuTensorMp. Integer and `Bool` inputs promote to `Float64` like other linalg.

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/contract.jl"]
```
