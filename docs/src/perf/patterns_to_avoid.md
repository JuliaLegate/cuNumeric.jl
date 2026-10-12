# Patterns to Avoid

By default behaviors which hinder performance, like scalar indexing,
converting a 0D NDArray to a Julia scalar, and implicit type promotion, are
disabled. The `@allow` macros permit users to selectively opt in to these
behaviors despite the potential performance loss. They are useful for
inspecting values while debugging, checking small test cases, or integrating external packages whose algorithms need
host-side decisions or type conversions.

| Macro | What it permits | Why it is guarded |
| :--- | :--- | :--- |
| `@allowscalar` | Accessing individual array elements, including in `for` loops | Repeated device-to-host transfers and possible communication across nodes |
| `@allowfetch` | Implicitly extracting device scalars for comparisons, predicates, or conversion to Julia numbers | The host must wait for the result |
| `@allowpromotion` | Implicit conversion of arrays to wider element types | Extra copies, memory use, and potentially slower arithmetic |

Wrap the smallest expression or block that needs the permission. Macro and
`do`-block forms restore the previous setting when the scope exits, including
on errors. Each permission is independent: `@allowfetch` does not enable
scalar indexing or wider type promotion.

For external packages, enable only the permissions required by the code path
being used. The [Krylov.jl example](../examples/krylov.md) uses `@allowfetch`
for convergence decisions and also `@allowpromotion` for complex CG.
[OrdinaryDiffEq.jl](../examples/ordinarydiffeq.md) can likewise encounter
host-side checks or generic array operations, depending on the solver and
options; the documented heat example uses settings that need no `@allow`
macros. Enabling a permission does not implement missing methods or establish
that an untested solver is compatible. Check correctness and performance for
your configuration.

## Scalar indexing

Accessing an `NDArray` one element at a time, such as `A[5]`, requires a
host value and can be expensive, especially in a loop. Prefer broadcasting,
reductions, and array operations. For a deliberate inspection while debugging:

```julia
using cuNumeric

A = cuNumeric.rand(Float32, 16)
value = @allowscalar A[5]
```

The equivalent scoped form is `allowscalar() do ... end`. Avoid enabling
scalar indexing for an entire program: it can hide slow element-by-element
fallbacks in otherwise generic Julia code.

## Fetching device scalars

Reductions such as `sum(A)` return a `CNScalar` that stays in runtime-managed
storage. Keep it there when using it in further arithmetic. Call `fetch`
when you deliberately need a native Julia number; it waits for the result
and needs neither `@allowfetch` nor `@allowscalar`.

```julia
s = sum(A)                  # device scalar
scaled = A .* s             # use it without fetching
host_sum = fetch(s)          # explicit host retrieval
```

Use `@allowfetch` when code implicitly needs a host value, such as a solver
comparing a residual with its stopping tolerance:

```julia
above_threshold = @allowfetch s > 1f0

host_sum = allowfetch() do
    Float32(s)
end
```

Arithmetic inside the scope still runs on the backend; values are fetched
only where host extraction is required. Repeated comparisons can repeatedly
synchronize, so keep them out of inner loops when array operations can express
the same calculation. `@allowfetch` does not automatically convert arguments
for arbitrary methods: a function requiring a `Float32` still needs an
explicit `Float32(s)` conversion.

See [Device Scalars](../api_cnscalar.md) and the
[TensorOperations.jl scalar example](../examples/tensoroperations.md#Scalars).

## Implicit promotion

Mixing different element types can widen an array, creating a copy and
increasing memory use. A common cause is mixing a `Float32` array with a
`Float64` literal: Julia's `1.0` is `Float64`, whereas `1f0` is `Float32`.
Match constants to the array's element type when wider precision is not needed:

```julia
B = A .+ 1f0                # keep Float32 arithmetic
C = @allowpromotion A .+ 1.0 # deliberately allow widening to Float64
```

Use `@allowpromotion` (or `allowpromotion() do ... end`) when the conversion
is intentional, including where an external algorithm mixes coefficient
types. The macro permits the conversion; it does not remove its allocation
or performance cost.
