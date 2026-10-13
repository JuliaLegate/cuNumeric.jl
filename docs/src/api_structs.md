# Struct Arrays

NDArrays hold structs as an array of structs (AoS) or, with StructArrays.jl, a
struct of arrays (SoA). Both need a CUDA device with broadcast fusion enabled.

## Array of structs

`NDArray{T}` where `T` is an `isbits` struct or `NamedTuple` of supported
scalar fields.

```julia
using cuNumeric

struct Particle
    position::Float32
    velocity::Float32
end
step(p, dt) = Particle(p.position + dt * p.velocity, p.velocity)
position(p) = p.position

particles = NDArray([Particle(0f0, 1f0), Particle(2f0, -0.5f0)])
particles .= step.(particles, 0.1f0)
positions = position.(particles)   # NDArray{Float32}
```

Not supported: nested structs or tuple fields, struct broadcast scalars,
size-1 extrusion, 0D broadcasts, and reductions.

## Struct of arrays

!!! warning "Experimental Feature"
    StructArray broadcasting requires `cuNumeric.Experimental(true)`.

Each field is an NDArray. One broadcast fills them all.

```julia
using cuNumeric, StructArrays
cuNumeric.Experimental(true)

struct Pair2
    a::Float64
    b::Float64
end
make_pair(x) = Pair2(x + 1.0, x + 2.0)

x = NDArray([0.0, 1.0, 2.0])
dest = StructArray{Pair2}((a=cuNumeric.zeros(Float64, 3), b=cuNumeric.zeros(Float64, 3)))
dest .= make_pair.(x)
```

Inputs must match the destination's shape. The right-hand side may read the
destination's own fields.
