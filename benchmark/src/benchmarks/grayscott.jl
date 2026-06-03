struct GSParams{T}
    dx::T
    dt::T
    c_u::T
    c_v::T
    f::T
    k::T
end

function GSParams{T}(; dx=1, c_u=1.0, c_v=0.3, f=0.03, k=0.06) where {T}
    GSParams{T}(T(dx), T(dx / 5), T(c_u), T(c_v), T(f), T(k))
end

Base.@kwdef struct GrayScott{T} <: AbstractBenchmark{T}
    N::Int
    M::Int
    variant::Symbol = :baseline
end

name(::GrayScott) = "grayscott"
dims(b::GrayScott) = (b.N, b.M)
data(b::GrayScott{T}) where {T} = "GrayScott with T=$(T), N=$(b.N), M=$(b.M)"
allowed_types(::Type{GrayScott}) = cuNumeric.SUPPORTED_FLOAT_TYPES
total_flops(b::GrayScott) = b.N * b.M # grid points updated per step

function build_benchmark(::Type{GrayScott}, ::Type{T}, N, M, variant) where {T}
    GrayScott{T}(; N=N, M=M, variant=Symbol(variant))
end

mutable struct GrayScottState{A,P}
    u::A
    v::A
    u_new::A
    v_new::A
    params::P
end

function initialize(b::GrayScott{T}) where {T}
    d = (b.N, b.M)
    u = cuNumeric.ones(T, d)
    v = cuNumeric.zeros(T, d)
    u_new = cuNumeric.zeros(T, d)
    v_new = cuNumeric.zeros(T, d)

    seed = min(150, b.N, b.M)
    u[1:seed, 1:seed] = cuNumeric.rand(T, (seed, seed))
    v[1:seed, 1:seed] = cuNumeric.rand(T, (seed, seed))

    return (GrayScottState(u, v, u_new, v_new, GSParams{T}()),)
end

# VARIANT DESCRIPTION
# baseline: as written
# lifetimes: step wrapped in @analyze_lifetimes
let body = quote
        # currently we don't have NDArray^x working yet.
        F_u = (
            (
                -u[2:(end - 1), 2:(end - 1)] .*
                (v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)])
            ) + args.f * (1 .- u[2:(end - 1), 2:(end - 1)])
        )
        F_v = (
            (
                u[2:(end - 1), 2:(end - 1)] .*
                (v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)])
            ) - (args.f + args.k) * v[2:(end - 1), 2:(end - 1)]
        )
        # 2-D Laplacian via slicing, excluding boundaries
        u_lap = (
            (
                u[3:end, 2:(end - 1)] - 2 * u[2:(end - 1), 2:(end - 1)] +
                u[1:(end - 2), 2:(end - 1)]
            ) ./ args.dx^2 +
            (
                u[2:(end - 1), 3:end] - 2 * u[2:(end - 1), 2:(end - 1)] +
                u[2:(end - 1), 1:(end - 2)]
            ) ./ args.dx^2
        )
        v_lap = (
            (
                v[3:end, 2:(end - 1)] - 2 * v[2:(end - 1), 2:(end - 1)] +
                v[1:(end - 2), 2:(end - 1)]
            ) ./ args.dx^2 +
            (
                v[2:(end - 1), 3:end] - 2 * v[2:(end - 1), 2:(end - 1)] +
                v[2:(end - 1), 1:(end - 2)]
            ) ./ args.dx^2
        )

        # Forward-Euler step for all interior points
        u_new[2:(end - 1), 2:(end - 1)] =
            ((args.c_u * u_lap) + F_u) * args.dt + u[2:(end - 1), 2:(end - 1)]
        v_new[2:(end - 1), 2:(end - 1)] =
            ((args.c_v * v_lap) + F_v) * args.dt + v[2:(end - 1), 2:(end - 1)]

        # Periodic boundary conditions
        u_new[:, 1] = u[:, end - 1]
        u_new[:, end] = u[:, 2]
        u_new[1, :] = u[end - 1, :]
        u_new[end, :] = u[2, :]
        v_new[:, 1] = v[:, end - 1]
        v_new[:, end] = v[:, 2]
        v_new[1, :] = v[end - 1, :]
        v_new[end, :] = v[2, :]
    end
    @eval _gs_step!(::Val{:baseline}, u, v, u_new, v_new, args::GSParams) = $body
    @eval _gs_step!(::Val{:lifetimes}, u, v, u_new, v_new, args::GSParams) = @analyze_lifetimes $body
end

# Variants not special-cased (e.g. testing fusion) run the baseline path.
function _gs_step!(::Val, u, v, u_new, v_new, args::GSParams)
    _gs_step!(Val(:baseline), u, v, u_new, v_new, args)
end

function run!(b::GrayScott, st::GrayScottState)
    _gs_step!(Val(b.variant), st.u, st.v, st.u_new, st.v_new, st.params)
    # swap references rather than copy
    st.u, st.u_new = st.u_new, st.u
    st.v, st.v_new = st.v_new, st.v
    return nothing
end

register_variant("lifetimes")
register_benchmark("grayscott", GrayScott)
