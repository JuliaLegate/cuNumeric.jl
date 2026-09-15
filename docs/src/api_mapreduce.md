# Mapped Reductions

`mapreduce(f, op, A; dims=:, init=...)` applies a scalar function and reduces its
results without allocating a mapped copy of `A`. Supported operators are `+`,
`*`, `min`, and `max`. The mapped forms of `sum`, `prod`, `minimum`, and `maximum`
use the same implementation.

```julia
A = cuNumeric.ones(Float32, 1024, 512)
energy = sum(abs2, A)                         # 0-d NDArray{Float32}
columns = mapreduce(abs2, +, A; dims=1)        # 1 × 512
total = mapreduce(abs2, +, A; dims=(1, 2))     # 1 × 1
α = 0.5f0
distance = sum(x -> abs2(x - α), A)
largest = maximum(abs, A; init=0f0)
```

Full reductions retain their result on the device as a 0-d NDArray. Explicit
dimensions preserve the array rank. Duplicate dimensions are ignored; positive
dimensions beyond the rank have no effect. `dims=()` still applies the mapping.

`sum(f, A)` and `prod(f, A)` widen small integer accumulators as Base does;
`mapreduce(f, +, A)` and `mapreduce(f, *, A)` use ordinary scalar arithmetic.
Implicit widening still requires `@allowpromotion`, including widening performed
by the mapping function. For example:

```julia
B = cuNumeric.ones(Int8, 16)
small = mapreduce(identity, +, B)             # Int8 accumulator
wide = @allowpromotion sum(identity, B)       # Int accumulator
```

## Initialization and empty inputs

Use a neutral scalar `init`, following Base's contract. It is applied once to the
combined result, never once per GPU partition. A full empty reduction returns
`init` when supplied. Without it, Base's empty-map rules apply: for example,
`sum(identity, A)` can produce zero for an empty array, whereas
`mapreduce(x -> x*x, +, A)` on an empty array requires `init`. Dimensional empty
sums/products produce their identities. Empty extrema reductions error, except
Base's special cases such as `maximum(abs2, A)`.

For full reductions, the final operation with `init` determines the output type.
For dimensional reductions, `typeof(init)` determines the output type. The latter
must also hold the inferred accumulator type without narrowing; combinations such
as a floating-point mapping with integer `init` are rejected before launching.
Use a matching or wider floating-point `init` instead. Narrowing after each
partial update is not an associative distributed operation.

Singleton reductions avoid reduction identities in the parallel kernels. Full
singletons follow Base's `reduce_first` behavior; dimensional sums/products still
apply Base's zero/one seed, which can matter for signed zeros and complex infinities.

## Current limitations

- The mapped API currently requires an active Legate GPU target. Existing unmapped
  reductions retain their CPU support. Broadcast-fusion settings do not control
  mapped reductions.
- One input NDArray is supported. Multiple arrays and custom binary reducers are
  rejected. Immutable scalar captures are supported; captured arrays, pointers,
  mutable captures, dynamic dispatch, and GPU-incompatible mapping functions are not.
- Mapping results must infer one concrete supported scalar type: Bool, the existing
  signed/unsigned integer widths, Float32/Float64, or ComplexF32/ComplexF64.
  Complex mapped values support addition/product only. Float16, strings, tuple
  accumulators, and union-valued results are unsupported.
- Parallel reassociation changes floating-point rounding, overflow behavior, and
  potentially signed zeros in arithmetic reductions. Bitwise agreement with Base
  or between different partitionings is not guaranteed. Floating-point extrema
  preserve NaN propagation and signed-zero ordering, but not NaN payloads.
- Mapping functions must be suitable for GPU compilation. Their calls must not
  rely on host side effects, allocation, or a particular evaluation order.
- PTX targets the Julia CUDA device used for compilation; every participating GPU
  must support that target. Heterogeneous GPU target selection is not implemented.

## Execution model

Legate partitions the input and provides a reduction store. Julia-generated PTX
maps each tile and reduces CUDA block partials into that store. Each retained
output coordinate has one writer within a task, using the pointer exposed by an
exclusive Legate reduction accessor (the same approach as cuPyNumeric's GEMV).
**Legate performs the reduction between tasks and devices.** The wrapper only
packs descriptors, allocates bounded task-local scratch, and launches PTX on the
task stream; it contains no native CUDA reduction kernels or custom all-reduce.

Scratch is bounded to 4,096 partial values per task. Floating-point extrema use
ordered integer keys and a small device decoding task. Kernel compilation and
registration happen on cache misses; warm calls enqueue work without extracting
host results or inserting execution fences.
When no decoding or initialization is needed, the accumulator is returned directly,
without a second output allocation or finishing task. Capture bytes are copied
into task-owned scalars at submission, and temporary NDArray handles are released
without waiting for GPU execution.

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/mapreduce.jl"]
```
