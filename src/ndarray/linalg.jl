function choose_nd_color_shape(shape::NTuple{N,Int}) where {N}
    color_shape = Base.ones(Int, N)
    if N > 2
        color_shape[1] = Legate.num_procs()
        done = false
        while !done && color_shape[1] % 2 == 0
            weight_per_dim = [shape[i] / color_shape[i] for i in 1:(N - 2)]
            max_weight, idx = findmax(weight_per_dim)
            if weight_per_dim[idx] > 2 * weight_per_dim[1]
                color_shape[1] ÷= 2
                color_shape[idx] *= 2
            else
                done = true
            end
        end
    end
    return Tuple(color_shape)
end

function prepare_manual_task_for_batched_matrices(full_shape::NTuple{N,Int}) where {N}
    initial_color_shape = choose_nd_color_shape(full_shape)
    tilesize = Tuple(
        (full_shape[i] + initial_color_shape[i] - 1) ÷ initial_color_shape[i] for i in 1:N
    )
    color_shape = Tuple((full_shape[i] + tilesize[i] - 1) ÷ tilesize[i] for i in 1:N)
    return tilesize, color_shape
end

function solve_batched(a::NDArray{T,N}, b::NDArray, x::NDArray) where {T,N}
    nrhs = size(b)[end]
    full_shape = size(a)
    tilesize_a, color_shape = prepare_manual_task_for_batched_matrices(full_shape)
    tilesize_b = (tilesize_a[1:(end - 1)]..., nrhs)

    store_a = nda_to_logical_store(a)
    store_b = nda_to_logical_store(b)
    store_x = nda_to_logical_store(x)

    tiled_a = Legate.partition_by_tiling(store_a, collect(tilesize_a))
    tiled_b = Legate.partition_by_tiling(store_b, collect(tilesize_b))
    tiled_x = Legate.partition_by_tiling(store_x, collect(tilesize_b))

    rt = Legate.get_runtime()
    domain = Legate.domain_from_shape(Legate.Shape(Legate.to_cxx_vector(color_shape)))
    lib = cuNumeric.get_lib()
    task = Legate.create_manual_task(rt, lib, cuNumeric.SOLVE, domain)

    Legate.add_input(task, tiled_a)
    Legate.add_input(task, tiled_b)
    Legate.add_output(task, tiled_x)

    Legate.submit_manual_task(rt, task)
end

# solve runs in floating point:
# int/bool inputs promote to Float64 (matching cupynumeric)
const _SOLVE_PROMOTABLE = Union{SUPPORTED_INT_TYPES,Bool}
const _SOLVE_ACCEPTED = Union{SUPPORTED_SOLVE_TYPES,_SOLVE_PROMOTABLE}
_solve_eltype(::Type{T}) where {T<:_SOLVE_PROMOTABLE} = Float64
_solve_eltype(::Type{T}) where {T<:SUPPORTED_SOLVE_TYPES} = T

# Type/dim guards dispatch on one argument at a time, then forward to `_solve`.
function solve(a::NDArray{<:_SOLVE_ACCEPTED}, b::NDArray{<:_SOLVE_ACCEPTED})
    A, B = eltype(a), eltype(b)
    O = promote_type(_solve_eltype(A), _solve_eltype(B))
    # int/bool -> float is an implicit promotion, disallowed unless `allowpromotion`
    A <: _SOLVE_PROMOTABLE && assertpromotion(solve, A, O)
    B <: _SOLVE_PROMOTABLE && assertpromotion(solve, B, O)
    return _solve_check_a_dims(unchecked_promote_arr(a, O), unchecked_promote_arr(b, O))
end

function solve(a::NDArray, b::NDArray)
    bad = eltype(a) <: _SOLVE_ACCEPTED ? eltype(b) : eltype(a)
    throw(ArgumentError("array type $bad is unsupported in solve"))
end

# `a` must be at least 2D, `b` at least 1D.
function _solve_check_a_dims(a::NDArray{<:Any,0}, b::NDArray)
    throw(ArgumentError("0-dimensional array given. Array must be at least two-dimensional"))
end
function _solve_check_a_dims(a::NDArray{<:Any,1}, b::NDArray)
    throw(ArgumentError("1-dimensional array given. Array must be at least two-dimensional"))
end
_solve_check_a_dims(a::NDArray, b::NDArray) = _solve_check_b_dims(a, b)

function _solve_check_b_dims(a::NDArray, b::NDArray{<:Any,0})
    throw(ArgumentError("0-dimensional array given. Array must be at least one-dimensional"))
end
_solve_check_b_dims(a::NDArray, b::NDArray) = _solve(a, b)

# 2D case: (m,m),(m)->(m).
# Backend needs rhs "b" to be 2D. We reshape b from (n,) to (n,1)
function _solve(a::NDArray{T,2}, b::NDArray{S,1}) where {T,S}
    m = size(b)[1]
    return reshape(_solve(a, reshape(b, (m, 1))), (m,))
end

# 2D (m,m),(m,n)->(m,n) and batched (...,m,m),(...,m,n)->(...,m,n)
function _solve(a::NDArray{T,N}, b::NDArray{S,N}) where {T,S,N}
    size(a)[end - 1] != size(a)[end] &&
        throw(ArgumentError("Last 2 dimensions of the array must be square"))
    size(a)[end] != size(b)[end - 1] &&
        throw(
            ArgumentError(
                "Input operand 1 has a mismatch in its dimension " *
                "$(N-2), with signature (...,m,m),(...,m,n)->(...,m,n)" *
                " (size $(size(b)[end-1]) is different from $(size(a)[end]))",
            ),
        )
    prod(size(a)) == 0 || prod(size(b)) == 0 && return zeros(T, size(b)...)
    x = zeros(T, size(b)...)
    solve_batched(a, b, x)
    return x
end

# Mismatched batch dimensions
function _solve(a::NDArray{T,N}, b::NDArray{S,M}) where {T,N,S,M}
    throw(ArgumentError("Batched matrices require signature (...,m,m),(...,m,n)->(...,m,n)"))
end

# ── svd ───────────────────────────────────────────────────────────────────────

function svd_single(a::NDArray{T,N}, u::NDArray, s::NDArray, vh::NDArray) where {T,N}
    rt  = Legate.get_runtime()
    lib = cuNumeric.get_lib()
    task = Legate.create_auto_task(rt, lib, cuNumeric.SVD)

    l_a  = nda_to_logical_array(a)
    l_u  = nda_to_logical_array(u)
    l_s  = nda_to_logical_array(s)
    l_vh = nda_to_logical_array(vh)

    Legate.add_input(task, l_a)
    Legate.add_output(task, l_u)
    Legate.add_output(task, l_s)
    Legate.add_output(task, l_vh)

    Legate.add_broadcast(task, l_a)
    Legate.add_broadcast(task, l_u)
    Legate.add_broadcast(task, l_s)
    Legate.add_broadcast(task, l_vh)

    Legate.submit_auto_task(rt, task)
end

function _svd(a::NDArray{T,2}, full_matrices::Bool) where {T}
    m, n = size(a)
    k = min(m, n)
    S = real(T)
    # cuSolver requires full square buffers regardless of full_matrices
    u_buf  = zeros(T, m, m)
    s      = zeros(S, k)
    vh_buf = zeros(T, n, n)
    svd_single(a, u_buf, s, vh_buf)
    # materialize transposed copies to fix cuSolver column-major layout
    u_full  = copy(cuNumeric.transpose(u_buf))
    vh_full = copy(cuNumeric.transpose(vh_buf))
    u  = full_matrices ? u_full  : u_full[:,  1:k]
    vh = full_matrices ? vh_full : vh_full[1:k, :]
    return u, s, vh
end

# svd runs on float/complex only — no integer backend
const _SVD_PROMOTABLE = Union{SUPPORTED_INT_TYPES,Bool}
const _SVD_ACCEPTED = Union{SUPPORTED_SVD_TYPES,_SVD_PROMOTABLE}
_svd_eltype(::Type{T}) where {T<:_SVD_PROMOTABLE} = Float64
_svd_eltype(::Type{T}) where {T<:SUPPORTED_SVD_TYPES} = T

function svd(a::NDArray{<:_SVD_ACCEPTED}, full_matrices::Bool=true)
    A = eltype(a)
    O = _svd_eltype(A)
    A <: _SVD_PROMOTABLE && assertpromotion(svd, A, O)
    return _svd_check_dims(unchecked_promote_arr(a, O), full_matrices)
end

function svd(a::NDArray, full_matrices::Bool=true)
    throw(ArgumentError("array type $(eltype(a)) is unsupported in svd"))
end

function _svd_check_dims(a::NDArray{<:Any,0}, full_matrices::Bool)
    throw(ArgumentError("0-dimensional array given. Array must be at least two-dimensional"))
end

function _svd_check_dims(a::NDArray{<:Any,1}, full_matrices::Bool)
    throw(ArgumentError("1-dimensional array given. Array must be at least two-dimensional"))
end

function _svd_check_dims(a::NDArray{<:Any,2}, full_matrices::Bool)
    return _svd(a, full_matrices)
end

function _svd_check_dims(a::NDArray, full_matrices::Bool)
    throw(ArgumentError("cuNumeric does not yet support stacked 2d arrays"))
end