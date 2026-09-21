# Device scalars and autounwrapping

Full reductions such as `sum(A)`, `dot(x, y)`, and `norm(x)` return an
`NDScalar`. Concrete wrappers mirror the numeric category of their storage:
`NDFloat <: AbstractFloat`, `NDInt <: Signed`, `NDUInt <: Unsigned`,
`NDBool <: Integer`, and `NDComplex <: Number`. `NDReal` is the union of the
four real wrapper families, and `NDScalar` also includes `NDComplex`.
Each owns a reference to a 0D NDArray, with no copy or host
extraction when it is wrapped. Reductions that retain dimensions still return
NDArrays. Explicit 0D array construction and array broadcasts remain arrays.

The wrappers retain both the element type `T` and the parent type `P` in a
concrete `value::NDArray{T,0,P}` field. `P` describes the storage owner, not
padding: an attached Julia 0D array can have a non-`Nothing` parent even when
it is unpadded. Constructors infer both parameters from the array.

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
They are read on the host because the current implementation uses them to select
algorithms or populate host runtime arguments. The `searchsorted` convenience
function still explicitly extracts its result indices to construct a Julia range;
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
`axpby!`, and TensorOperations calls without enabling autounwrap. These paths
use their backing 0D arrays. Zero/one shortcuts inspect host coefficients only;
a device coefficient's value is not read to choose a shortcut.

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

## Performance without autounwrapping

These are implementation-level costs, not benchmark results. Disabling
autounwrap prevents implicit extraction; it does not disable device scalar
wrapping or backend work. Permission can also be enabled by the do-block or
`autounwrap(true)`, independently of whether a macro appears in the code.

| Operation | Cost or behavior with extraction permission disabled |
|:--|:--|
| Existing calls with ordinary host numbers | The scalar normalization helpers return the host value or use the existing host path. They do not inspect task-local permission or add device work. Additional Julia forwarding calls should generally specialize away, but zero runtime overhead has not been benchmarked. |
| Full reductions | Construct an immutable numeric wrapper around the existing 0D result. Wrapping adds no backend allocation, kernel, or synchronization. The Julia-side wrapper may be optimized away; allocation-free execution is not guaranteed in every calling context. |
| Scalar arithmetic | Uses backend broadcasts and result arrays, with no autounwrap permission lookup. Separate non-dotted operations such as `a*b+c` submit separate operations; the wrapper does not fuse them into one kernel. This is more expensive than computing with already-materialized host scalars. |
| Scalar unary operations | `_scalar_unary` creates size-one and rank-zero reshape handles around a broadcast. Composite operations such as complex `abs2` currently perform several backend operations. These are optimization opportunities, independent of the permission setting. |
| `zero(s)` / `one(s)` | Construct backend scalar storage, rather than a cheap native numeric constant. Generic numeric code can therefore incur allocations and task submissions here. |
| Device coefficients and fill values | Use array/broadcast paths. Device coefficients do not take value-dependent host zero/one shortcuts, so they can require extra work or temporaries compared with equivalent host constants. `fill(device_value, dims)` currently allocates zeros and then broadcasts the fill. Existing host coefficient/fill paths retain their shortcuts. |
| Implicit comparisons, predicates, and checked host conversions | Perform a task-local permission lookup and then throw when disabled. They do not silently synchronize and continue. |
| Explicit `unwrap` / `only`, or display | Extract host values without autounwrap permission. REPL display therefore synchronizes unless suppressed with `;`, matching the previous 0D NDArray display behavior. |
| Loading and first use | Extra numeric methods and wrapper specializations can increase precompilation, compilation, and code size. These costs have not been measured. |

The comparison for backend arithmetic is important: a wrapper around a 0D array
does not intrinsically add a kernel, but choosing device scalar arithmetic over
native host arithmetic does add backend work. Explicitly extracting a scalar
may make subsequent arithmetic cheaper, at the cost of synchronization.

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
