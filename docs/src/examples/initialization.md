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

# Uninitialized: write every element before reading
U = NDArray{Float32}(undef, 4, 4)
S = similar(U)                            # same type and shape, also uninitialized

# Boolean arrays
T = cuNumeric.trues(2, 3)
Fbool = cuNumeric.falses(2, 3)

# Identity: prefer Diagonal / I; densify only when needed (no eye)
D = Diagonal(cuNumeric.ones(Float32, 5))
I32 = NDArray{Float32}(LinearAlgebra.I, 5, 5)   # dense identity
Ib = NDArray(LinearAlgebra.I, 5, 5)             # Bool identity

# Uniform / normal random values (native Float32 and Float64)
R = cuNumeric.rand(4, 4)
R64 = cuNumeric.rand(Float64, 1000)
cuNumeric.rand!(R64)                      # fill an existing Float64 array
N = cuNumeric.randn(Float32, 8, 8)
C = cuNumeric.rand(ComplexF32, 8, 8)      # independent real/imag uniforms
E = cuNumeric.randexp(Float64, 1000)      # exponential, scale 1 (real only)
Ints = cuNumeric.rand(0:9, 4, 4)             # Int64 in 0:9 (inclusive)
Coin = cuNumeric.rand(Bool, 8)            # fair coin flips

# Private stream / non-default engine (see RNG in the Public API)
g = cuNumeric.default_rng(cuNumeric.PHILOX4_32_10, 1234)
P = cuNumeric.random(g, Float32, (4, 4))
```

Shapes can be passed as separate `Int`s or as a `Tuple` / `Dims`:

```julia
cuNumeric.zeros(2, 3)
cuNumeric.zeros((2, 3))
cuNumeric.ones(Float64, (10, 10))
```

For signatures and more detail, see [Initialization](../api_initialization.md) and [RNG](../api_random.md) in the Public API.
