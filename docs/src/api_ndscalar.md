# Device scalars and autounwrapping

Device scalars let you use reduction results in further calculations without
first copying their values to the host. They also participate in Julia's numeric
type hierarchy, so they can be passed to compatible numeric methods and stored
in structs with abstract numeric type constraints. Extract a host value explicitly
when you need one, or enable scoped autounwrapping for comparisons and conversions.

Full reductions such as `sum(A)`, `dot(x, y)`, and `norm(x)` return an
`NDScalar`. Concrete wrappers mirror the numeric category of their storage:
`NDFloat <: AbstractFloat`, `NDInt <: Signed`, `NDUInt <: Unsigned`,
`NDBool <: Integer`, and `NDComplex <: Number`. `NDReal` is the union of the
four real wrapper families, and `NDScalar` also includes `NDComplex`.
Each owns a reference to a 0D NDArray, with no copy or host
extraction when it is wrapped. Reductions that retain dimensions still return
NDArrays. Explicit 0D array construction and array broadcasts remain arrays.

`DeviceScalar{T}` is the shared dispatch alias for a raw `NDArray{T,0}` or an
`NDScalar{T}` wrapper. Use `DeviceScalar` when either representation is accepted:

```julia
twice(x::DeviceScalar) = x .* 2
```

Numeric data arguments such as search needles, fill values, and
linear algebra coefficients use this shared storage path. `Ref(s)` in broadcast
also preserves device storage for either representation.

Host control parameters, including a norm's `p`, random distribution parameters,
and `isapprox` tolerances, require autounwrap permission for either representation.
The `searchsorted` convenience function explicitly extracts its result indices
to construct a Julia range;
use `searchsortedfirst`/`searchsortedlast` to retain device results.

```julia
using cuNumeric, LinearAlgebra
x = cuNumeric.NDArray([1.0, 2.0, 3.0])
s = sum(x)                    # NDFloat{Float64}, accepted in <:AbstractFloat fields
t = s^2 / 2                   # another device scalar
y = x .* t                    # backend broadcast; no implicit host extraction
value = unwrap(t)             # explicit synchronization, always permitted
```

`ndscalar(a)` wraps an existing 0D NDArray; `s.value` accesses its backend array
without synchronizing. Use `unwrap(s)` or `only(s)` for explicit host extraction.
Arithmetic, `min`/`max`, and promotion between numeric types keep results on the
backend. Showing a wrapper uses the backing 0D NDArray's display and extracts
its value, including in the REPL, without requiring autounwrap permission. End
an expression with `;` to suppress REPL display and that synchronization.
The existing promotion policy still applies.

Device scalars can also be coefficients in `contract!`, `mul!`, `axpy!`,
`axpby!`, and TensorOperations calls without enabling autounwrap.

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

The permission also applies inside ordinary functions called within the scope;
the macro does not need to inspect their bodies:

```julia
function host_residual(x)
    r = norm(x)                # device scalar
    return Float64(r)          # permission checked here, inside this function
end

residual = @autounwrap host_residual(x)
# host_residual(x)             # errors outside the scope
```

Arithmetic remains asynchronous even inside an autounwrap scope. There is no
general fallback that unwraps arguments when Julia cannot find a method.
For example, a function accepting only `Float64` still needs `f(Float64(s))`.
Julia also requires an actual Bool in `if`: use `Bool(all(A))` within the scope,
or a comparison that returns a host Bool. Autounwrapping does not change Julia's
dispatch or condition evaluation rules.
