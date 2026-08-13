# Initialization

Create `NDArray`s with the usual Julia-style constructors. The default element type is `Float32` unless you pass one.

```julia
using cuNumeric

# Zeros / ones / fill
Z = cuNumeric.zeros(4, 4)                 # Float32 by default
Z64 = cuNumeric.zeros(Float64, 4, 4)
O = cuNumeric.ones(3, 3)
F = cuNumeric.fill(7.5f0, (2, 3))

# Boolean arrays
T = cuNumeric.trues(2, 3)
Fbool = cuNumeric.falses(2, 3)

# Identity
I = cuNumeric.eye(5)
I16 = cuNumeric.eye(Float32, 5)

# Uniform / normal random values (native Float32 and Float64)
R = cuNumeric.rand(4, 4)
R64 = cuNumeric.rand(Float64, 1000)
cuNumeric.rand!(R64)                      # fill an existing Float64 array
N = cuNumeric.randn(Float32, 8, 8)
E = cuNumeric.randexp(Float64, 1000)      # exponential, scale 1
I = cuNumeric.rand(0:9, 4, 4)             # Int64 in 0:9 (inclusive)
Coin = cuNumeric.rand(Bool, 8)            # fair coin flips

# Private stream / non-default engine (see Random in the Public API)
g = cuNumeric.default_rng(cuNumeric.PHILOX4_32_10, 1234)
P = cuNumeric.random(g, Float32, (4, 4))
```

Shapes can be passed as separate `Int`s or as a `Tuple` / `Dims`:

```julia
cuNumeric.zeros(2, 3)
cuNumeric.zeros((2, 3))
cuNumeric.ones(Float64, (10, 10))
```

For signatures and more detail, see [Initialization](../api_initialization.md) and [Random](../api_random.md) in the Public API.
