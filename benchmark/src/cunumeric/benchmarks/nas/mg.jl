# LIMITATION: NPB MG's exact sparse right-hand side is generated on the host,
# as in NPB-GPU, then uploaded once before timing. The V-cycle itself uses
# cuNumeric views and broadcasts, allowing Legate to partition every level.
# Restriction/interpolation are separable axis passes with temporary arrays,
# not JACC's direct per-cell kernels. Views are no-ops; explicit slice/store
# operations express the regions. The common harness times initial zeroing
# and L2 sum-of-squares, but omits NPB's Linf norm; see nas/README.md.

mutable struct CuNumericNASMGState{U,R,V,W}
    u::U
    r::R
    rhs::V
    interp_weights::W
end

const CuNumericMGIndex = Union{Colon,AbstractUnitRange{<:Integer}}

function cunumeric_mg_slice(index::Colon, n)
    return (0, n)
end
cunumeric_mg_slice(index::AbstractUnitRange, n) = (Int(first(index)) - 1, Int(last(index)))

# cuNumeric's core currently defines contiguous slicing through rank 2. MG is
# rank 3, so provide the identical view operation locally until core generalizes
# those methods. The returned NDArray shares its parent's Legate store.
function Base.getindex(
    array::cuNumeric.NDArray{T,3}, i::CuNumericMGIndex,
    j::CuNumericMGIndex, k::CuNumericMGIndex,
) where {T}
    @boundscheck checkbounds(array, i, j, k)
    slices = cuNumeric.slice_array(
        cunumeric_mg_slice(i, size(array, 1)),
        cunumeric_mg_slice(j, size(array, 2)),
        cunumeric_mg_slice(k, size(array, 3)),
    )
    return cuNumeric.nda_get_slice(array, slices)
end

function Base.setindex!(
    lhs::cuNumeric.NDArray{T,3}, rhs::cuNumeric.NDArray,
    i::CuNumericMGIndex, j::CuNumericMGIndex, k::CuNumericMGIndex,
) where {T}
    view = lhs[i, j, k]
    copyto!(view, rhs)
    cuNumeric.destroy!(view)
    return rhs
end

function initialize(b::NASMultiGrid{Float64}; mod=cuNumeric)
    p = validate_nas_mg(b)
    sizes = nas_mg_level_sizes(p)
    u = [mod.zeros(Float64, n, n, n) for n in sizes]
    r = [mod.zeros(Float64, n, n, n) for n in sizes]
    rhs = mod.NDArray(nas_mg_rhs(p))
    weights = mod.reshape(mod.NDArray([0.0, 0.5]), 1, 1, 1, 2)
    return (CuNumericNASMGState(u, r, rhs, weights),)
end

function cunumeric_mg_back(array, axis)
    axis == 3 && return array, (1, 2, 3)
    permutation = axis == 1 ? (2, 3, 1) : (1, 3, 2)
    return permutedims(array, permutation), invperm(permutation)
end

function cunumeric_mg_restrict_axis(array, axis)
    back, inverse = cunumeric_mg_back(array, axis)
    d1, d2, n = size(back)
    physical = n - 2
    # cuNumeric reshapes stores in C order, so move the active dimension to
    # the back before pairing neighboring points. Legate cannot reshape a
    # sliced store; materialize only these two contiguous shifted slabs.
    left = copy(back[:, :, 2:(n - 1)])
    right = copy(back[:, :, 3:n])
    paired_shape = (d1, d2, physical ÷ 2, 2)
    reduced = cuNumeric.reshape(
        sum(cuNumeric.reshape(left, paired_shape); dims=4) .+
        sum(cuNumeric.reshape(right, paired_shape); dims=4),
        d1, d2, physical ÷ 2,
    )
    return axis == 3 ? reduced : permutedims(reduced, inverse)
end

function cunumeric_mg_restrict!(coarse, fine)
    reduced = cunumeric_mg_restrict_axis(fine, 1)
    reduced = cunumeric_mg_restrict_axis(reduced, 2)
    reduced = cunumeric_mg_restrict_axis(reduced, 3)
    n = size(coarse, 1)
    interior = coarse[2:(n - 1), 2:(n - 1), 2:(n - 1)]
    interior .= reduced ./ 16.0
    cuNumeric.destroy!(interior)
    return nas_mg_comm3!(coarse)
end

function cunumeric_mg_interp_axis(array, axis, weights)
    back, inverse = cunumeric_mg_back(array, axis)
    d1, d2, n = size(back)
    lo = copy(back[:, :, 1:(n - 1)])
    hi = copy(back[:, :, 2:n])
    lo4 = cuNumeric.reshape(lo, d1, d2, n - 1, 1)
    hi4 = cuNumeric.reshape(hi, d1, d2, n - 1, 1)
    mixed = lo4 .+ weights .* (hi4 .- lo4)
    interpolated = cuNumeric.reshape(mixed, d1, d2, 2(n - 1))
    result = axis == 3 ? interpolated : permutedims(interpolated, inverse)
    # The next interpolation axis permutes this result. Give it an independent
    # store instead of composing a transpose with the reshape view above.
    return copy(result)
end

function cunumeric_mg_interp!(fine, coarse, weights)
    interpolated = cunumeric_mg_interp_axis(coarse, 1, weights)
    interpolated = cunumeric_mg_interp_axis(interpolated, 2, weights)
    interpolated = cunumeric_mg_interp_axis(interpolated, 3, weights)
    fine .+= interpolated
    return fine
end

function cunumeric_mg_cycle!(s, c)
    finest = length(s.u)
    for level in finest:-1:2
        cunumeric_mg_restrict!(s.r[level - 1], s.r[level])
    end
    fill!(s.u[1], 0.0)
    nas_mg_psinv!(s.u[1], s.r[1], c)
    for level in 2:(finest - 1)
        fill!(s.u[level], 0.0)
        cunumeric_mg_interp!(s.u[level], s.u[level - 1], s.interp_weights)
        nas_mg_resid!(s.r[level], s.u[level], s.r[level])
        nas_mg_psinv!(s.u[level], s.r[level], c)
    end
    cunumeric_mg_interp!(s.u[end], s.u[end - 1], s.interp_weights)
    nas_mg_resid!(s.r[end], s.u[end], s.rhs)
    nas_mg_psinv!(s.u[end], s.r[end], c)
    return nothing
end

function cunumeric_mg_norm2(residual)
    n = size(residual, 1)
    interior = residual[2:(n - 1), 2:(n - 1), 2:(n - 1)]
    squared = sum(interior .* interior)
    cuNumeric.destroy!(interior)
    return squared
end

function run!(b::NASMultiGrid, s::CuNumericNASMGState)
    p = nas_mg_parameters(b.class)
    c = nas_mg_smoother(b.class)
    foreach(x -> fill!(x, 0.0), s.u)
    nas_mg_resid!(s.r[end], s.u[end], s.rhs)
    cunumeric_mg_norm2(s.r[end])
    for _ in 1:p.niter
        cunumeric_mg_cycle!(s, c)
        nas_mg_resid!(s.r[end], s.u[end], s.rhs)
    end
    return cunumeric_mg_norm2(s.r[end])
end

function check_benchmark_correctness(b::NASMultiGrid, gs::GlobalSettings; mod=cuNumeric)
    state = only(initialize(b; mod))
    squared = run!(b, state)
    norm = sqrt(cuNumeric.@allowscalar squared[] / Float64(b.N)^3)
    return nas_mg_verified(b.class, norm) ? "pass" : "fail"
end
