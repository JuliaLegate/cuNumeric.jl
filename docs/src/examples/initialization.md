# Initialization

Create `NDArray`s with the usual Julia-style constructors. The default element type is `Float32` unless you pass one.

```julia
using LinearAlgebra
using cuNumeric

# Zeros / ones / fill
Z = cuNumeric.zeros(4, 4)                 # Float32 by default
Z64 = cuNumeric.zeros(Float64, 4, 4)
O = cuNumeric.ones(3, 3)
F = cuNumeric.fill(7.5f0, (2, 3))

# Boolean arrays
T = cuNumeric.trues(2, 3)
Fbool = cuNumeric.falses(2, 3)

# Identity: prefer Diagonal / I; densify only when needed (no eye)
D = Diagonal(cuNumeric.ones(Float32, 5))
I32 = NDArray{Float32}(I, 5, 5)   # dense identity
Ib = NDArray(I, 5, 5)             # Bool identity

# Uniform random values (default Float32; backend draws Float64 then converts)
R = cuNumeric.rand(4, 4)
R64 = cuNumeric.rand(Float64, 1000)
cuNumeric.rand!(R64)                      # fill an existing Float64 array
```

Shapes can be passed as separate `Int`s or as a `Tuple` / `Dims`:

```julia
cuNumeric.zeros(2, 3)
cuNumeric.zeros((2, 3))
cuNumeric.ones(Float64, (10, 10))
```

For signatures and more detail, see [Initialization](../api_initialization.md) in the Public API.
