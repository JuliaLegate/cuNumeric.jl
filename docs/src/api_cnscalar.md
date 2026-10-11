# Device Scalars

Full reductions (`sum(A)`, `dot(x, y)`, `norm(x)`) return a `CNScalar`: a
number that stays on the device, backed by a 0-d NDArray. Using it in further
math launches no copy and no synchronization.

```julia
using cuNumeric, LinearAlgebra
x = cuNumeric.NDArray([1.0, 2.0, 3.0])
s = sum(x)        # CNFloat{Float64}
t = s^2 / 2       # still on device
y = x .* t        # no host round trip
fetch(t)          # 18.0, a Julia Float64
```

## Types

| Type | Subtype of |
|---|---|
| `CNFloat` | `AbstractFloat` |
| `CNInt`, `CNUInt` | `Signed`, `Unsigned` |
| `CNBool` | `Integer` |
| `CNComplex` | `Number` |

`CNReal` is the union of the real types, and `CNScalar` adds `CNComplex`.
`DeviceScalar{T}` accepts either a `CNScalar{T}` or an `NDArray{T,0}`.
`cnscalar(a)` wraps a 0-d NDArray.

## Getting a host value

`fetch(s)` always works. Implicit extraction (comparisons, `Float64(s)`,
`if` conditions, `isapprox` tolerances, a norm's `p`) needs `allowfetch`:

```julia
@allowfetch s > 0            # Bool
allowfetch() do
    Float64(s)
end
```

- The permission is task-local and also applies inside functions called in the
  scope.
- It doesn't convert arguments for you: a method that only takes `Float64` still
  needs `f(Float64(s))`.
- Conversions that can lose information (float to int, complex to real) need
  `allowfetch` and throw `InexactError` like Base.

Without `allowfetch`, device scalars work as broadcast operands, `Ref(s)`,
fill values, and coefficients for `mul!`, `axpy!`, `axpby!`, `contract!` and
TensorOperations.

```@autodocs
Modules = [cuNumeric]
Pages = ["cnscalar.jl"]
```
