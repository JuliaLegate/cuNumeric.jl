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

# One batch dimension is the ceiling for every batched op:
#   - the POTRF task body is only instantiated for 2 <= DIM < 4
#   - Legate.jl's `domain_from_shape` builds launch domains for at most three
#     dimensions and yields an empty domain past that
const MAX_BATCHED_DIM = 3

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

    @task_scope "solve" begin
        rt = Legate.get_runtime()
        domain = Legate.domain_from_shape(Legate.Shape(Legate.to_cxx_vector(color_shape)))
        lib = cuNumeric.get_lib()
        task = Legate.create_manual_task(rt, lib, cuNumeric.SOLVE, domain)
        cuNumeric.task_throws_exception(task, true)

        Legate.add_input(task, tiled_a)
        Legate.add_input(task, tiled_b)
        Legate.add_output(task, tiled_x)

        Legate.submit_manual_task(rt, task)
    end
end

# solve runs in floating point:
# int/bool inputs promote to Float64 (matching cupynumeric)
const _SOLVE_PROMOTABLE = Union{SUPPORTED_INT_TYPES,Bool}
const _SOLVE_ACCEPTED = Union{SUPPORTED_SOLVE_TYPES,_SOLVE_PROMOTABLE}
_solve_eltype(::Type{T}) where {T<:_SOLVE_PROMOTABLE} = Float64
_solve_eltype(::Type{T}) where {T<:SUPPORTED_SOLVE_TYPES} = T

# `a` must be at least 2D, `b` at least 1D. `solve` takes only the 2D case;
# stacked systems go through `batched_solve`.
function _solve_check_a_dims_2d(a::NDArray{<:Any,0}, b::NDArray)
    return throw(ArgumentError("0-dimensional array given. Array must be two-dimensional"))
end
function _solve_check_a_dims_2d(a::NDArray{<:Any,1}, b::NDArray)
    return throw(ArgumentError("1-dimensional array given. Array must be two-dimensional"))
end
_solve_check_a_dims_2d(a::NDArray{<:Any,2}, b::NDArray) = _solve_check_b_dims(a, b)
function _solve_check_a_dims_2d(a::NDArray{<:Any,N}, b::NDArray) where {N}
    return throw(
        ArgumentError(
            "$N-dimensional array given. Use `cuNumeric.batched_solve` for " *
            "stacked systems of shape (...,m,m)",
        ),
    )
end

function _solve_check_a_dims_batched(a::NDArray{<:Any,N}, b::NDArray) where {N}
    _assert_batched_dims(:batched_solve, N)
    return _solve_check_b_dims(a, b)
end

function _assert_batched_dims(f, N::Integer)
    N < 3 && throw(
        ArgumentError(
            "$N-dimensional array given. `$f` requires shape (b,m,m); use the " *
            "matching `LinearAlgebra` or `cuNumeric` function for a single matrix",
        ),
    )
    N > MAX_BATCHED_DIM && throw(
        ArgumentError(
            "$N-dimensional array given. `$f` supports at most one batch " *
            "dimension, i.e. shape (b,m,m)",
        ),
    )
    return nothing
end

function _solve_check_b_dims(a::NDArray, b::NDArray{<:Any,0})
    return throw(ArgumentError("0-dimensional array given. Array must be at least one-dimensional"))
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
    prod(size(a)) == 0 || prod(size(b)) == 0 && return cuNumeric.zeros(T, size(b)...)
    x = cuNumeric.zeros(T, size(b)...)
    solve_batched(a, b, x)
    return x
end

# Mismatched batch dimensions
function _solve(a::NDArray{T,N}, b::NDArray{S,M}) where {T,N,S,M}
    return throw(ArgumentError("Batched matrices require signature (...,m,m),(...,m,n)->(...,m,n)"))
end

# cholesky

"""
    potrf!(out, a; lower, zeroout)

Run the cupynumeric `POTRF` task, writing the Cholesky factor of `a` into `out`.

The task is batched over the leading dimension: for a `(b, m, m)` input it
factors each `m × m` block independently. `lower` selects which triangle holds
the factor, `zeroout` zeros the opposite triangle in-task.
"""
function potrf!(out::NDArray{T,N}, a::NDArray{T,N}; lower::Bool, zeroout::Bool) where {T,N}
    rt = Legate.get_runtime()
    lib = cuNumeric.get_lib()

    @task_scope "potrf" begin
        task = Legate.create_auto_task(rt, lib, cuNumeric.POTRF)
        cuNumeric.task_throws_exception(task, true)

        l_a = nda_to_logical_array(a)
        l_out = nda_to_logical_array(out)

        in_var = Legate.add_input(task, l_a)
        out_var = Legate.add_output(task, l_out)

        Legate.add_scalar(task, Legate.Scalar(lower))
        Legate.add_scalar(task, Legate.Scalar(zeroout))

        # Each matrix must live on one processor; only the batch axes may split.
        Legate.add_broadcast(task, l_a, CxxWrap.StdVector(UInt32[N - 2, N - 1]))
        Legate.add_constraint(task, Legate.align(out_var, in_var))

        Legate.submit_auto_task(rt, task)
    end
    return out
end

# eigen

"""
    geev!(a, ew, ev)

Run the cupynumeric `GEEV` task, writing eigenvalues into `ew` and, unless `ev`
is `nothing`, right eigenvectors into `ev` as columns.

`ew` has one fewer dimension than `a`, so the launch domain maps onto it through
a projection that drops the trailing axis. The task keys eigenvector computation
off the number of registered outputs, so `ev` must not be added when only
eigenvalues are wanted.
"""
function geev!(a::NDArray{T,N}, ew::NDArray, ev::Union{NDArray,Nothing}) where {T,N}
    full_shape = size(a)
    tilesize, color_shape = prepare_manual_task_for_batched_matrices(full_shape)

    tiled_a = Legate.partition_by_tiling(nda_to_logical_store(a), collect(tilesize))
    tiled_ew = Legate.partition_by_tiling(
        nda_to_logical_store(ew), collect(tilesize[1:(end - 1)])
    )
    tiled_ev = if ev === nothing
        nothing
    else
        Legate.partition_by_tiling(nda_to_logical_store(ev), collect(tilesize))
    end

    # 0-based source dimensions of the launch domain.
    proj = CxxWrap.StdVector(Int32.(0:(N - 1)))
    proj_ew = CxxWrap.StdVector(Int32.(0:(N - 2)))

    @task_scope "geev" begin
        rt = Legate.get_runtime()
        domain = Legate.domain_from_shape(Legate.Shape(Legate.to_cxx_vector(color_shape)))
        lib = cuNumeric.get_lib()
        task = Legate.create_manual_task(rt, lib, cuNumeric.GEEV, domain)
        cuNumeric.task_throws_exception(task, true)

        cuNumeric.add_input_proj(task, tiled_a.handle, proj)
        cuNumeric.add_output_proj(task, tiled_ew.handle, proj_ew)
        if tiled_ev !== nothing
            cuNumeric.add_output_proj(task, tiled_ev.handle, proj)
        end

        Legate.submit_manual_task(rt, task)
    end
    return nothing
end

function svd_single(a::NDArray{T,N}, u::NDArray, s::NDArray, vh::NDArray) where {T,N}
    rt = Legate.get_runtime()
    lib = cuNumeric.get_lib()
    task = Legate.create_auto_task(rt, lib, cuNumeric.SVD)

    l_a = nda_to_logical_array(a)
    l_u = nda_to_logical_array(u)
    l_s = nda_to_logical_array(s)
    l_vh = nda_to_logical_array(vh)

    Legate.add_input(task, l_a)
    Legate.add_output(task, l_u)
    Legate.add_output(task, l_s)
    Legate.add_output(task, l_vh)

    Legate.add_broadcast(task, l_a)
    Legate.add_broadcast(task, l_u)
    Legate.add_broadcast(task, l_s)
    Legate.add_broadcast(task, l_vh)

    return Legate.submit_auto_task(rt, task)
end

function _svd(a::NDArray{T,2}, full_matrices::Bool) where {T}
    m, n = size(a)
    k = min(m, n)
    S = real(T)
    # cuSolver requires full square buffers regardless of full_matrices
    u_buf = cuNumeric.zeros(T, m, m)
    s = cuNumeric.zeros(S, k)
    vh_buf = cuNumeric.zeros(T, n, n)
    svd_single(a, u_buf, s, vh_buf)
    # Backend factors are logically ordered; only thin strided views need materialization.
    u = full_matrices ? u_buf : copy(u_buf[:, 1:k])
    vh = full_matrices ? vh_buf : copy(vh_buf[1:k, :])
    return u, s, vh
end

# svd runs on float/complex only — no integer backend
const _SVD_PROMOTABLE = Union{SUPPORTED_INT_TYPES,Bool}
const _SVD_ACCEPTED = Union{SUPPORTED_SVD_TYPES,_SVD_PROMOTABLE}
_svd_eltype(::Type{T}) where {T<:_SVD_PROMOTABLE} = Float64
_svd_eltype(::Type{T}) where {T<:SUPPORTED_SVD_TYPES} = T

# qr

function qr_single(a::NDArray{T,N}, q::NDArray, r::NDArray) where {T,N}
    rt = Legate.get_runtime()
    lib = cuNumeric.get_lib()
    task = Legate.create_auto_task(rt, lib, cuNumeric.CQR)

    l_a = nda_to_logical_array(a)
    l_q = nda_to_logical_array(q)
    l_r = nda_to_logical_array(r)

    Legate.add_input(task, l_a)
    Legate.add_output(task, l_q)
    Legate.add_output(task, l_r)

    Legate.add_broadcast(task, l_a)
    Legate.add_broadcast(task, l_q)
    Legate.add_broadcast(task, l_r)

    return Legate.submit_auto_task(rt, task)
end

function _qr(a::NDArray{T,2}) where {T}
    m, n = size(a)
    k = min(m, n)
    # cuSolver requires full square buffers regardless of output shape
    q_buf = cuNumeric.zeros(T, m, m)
    r_buf = cuNumeric.zeros(T, n, n)
    qr_single(a, q_buf, r_buf)
    # Host conversion assumes contiguous storage, so materialize the economy slices.
    q = copy(q_buf[:, 1:k])
    r = copy(r_buf[1:k, :])
    return q, r
end

const _QR_PROMOTABLE = Union{SUPPORTED_INT_TYPES,Bool}
const _QR_ACCEPTED = Union{SUPPORTED_QR_TYPES,_QR_PROMOTABLE}
_qr_eltype(::Type{T}) where {T<:_QR_PROMOTABLE} = Float64
_qr_eltype(::Type{T}) where {T<:SUPPORTED_QR_TYPES} = T

# cholesky/eigen guards. Both run in floating point only, so int/bool inputs
# promote to Float64 the same way solve/svd/qr do.

const _CHOLESKY_PROMOTABLE = Union{SUPPORTED_INT_TYPES,Bool}
const _CHOLESKY_ACCEPTED = Union{SUPPORTED_CHOLESKY_TYPES,_CHOLESKY_PROMOTABLE}
_cholesky_eltype(::Type{T}) where {T<:_CHOLESKY_PROMOTABLE} = Float64
_cholesky_eltype(::Type{T}) where {T<:SUPPORTED_CHOLESKY_TYPES} = T

const _EIG_PROMOTABLE = Union{SUPPORTED_INT_TYPES,Bool}
const _EIG_ACCEPTED = Union{SUPPORTED_EIG_TYPES,_EIG_PROMOTABLE}
_eig_eltype(::Type{T}) where {T<:_EIG_PROMOTABLE} = Float64
_eig_eltype(::Type{T}) where {T<:SUPPORTED_EIG_TYPES} = T

# GEEV always produces complex eigenvalues and eigenvectors, even for real input.
_eig_complex_eltype(::Type{Float32}) = ComplexF32
_eig_complex_eltype(::Type{ComplexF32}) = ComplexF32
_eig_complex_eltype(::Type{Float64}) = ComplexF64
_eig_complex_eltype(::Type{ComplexF64}) = ComplexF64

function _check_square_matrices(f, a::NDArray{<:Any,N}) where {N}
    N < 2 && throw(
        ArgumentError(
            "$N-dimensional array given. Array must be at least two-dimensional"
        ),
    )
    sz = size(a)
    sz[end - 1] != sz[end] &&
        throw(ArgumentError("Last 2 dimensions of the array must be square in $f"))
    sz[end] == 0 && throw(ArgumentError("Input shape dimension 0 not allowed in $f"))
    return nothing
end

"""
    _cholesky(a)

Lower Cholesky factor of each trailing square block of `a`, with the upper
triangle zeroed. Only the lower triangle of the input is read; the input is
assumed Hermitian without being checked, matching cupynumeric.
"""
function _cholesky(a::NDArray{T,N}) where {T,N}
    _check_square_matrices(:cholesky, a)
    out = cuNumeric.zeros(T, size(a)...)
    potrf!(out, a; lower=true, zeroout=true)
    return out
end

"""
    _eig(a)

Eigenvalues and right eigenvectors of each trailing square block of `a`. Both
are always complex.
"""
function _eig(a::NDArray{T,N}) where {T,N}
    ew = _alloc_eigenvalues(a)
    ev = cuNumeric.zeros(_eig_complex_eltype(T), size(a)...)
    geev!(a, ew, ev)
    return ew, ev
end

"""
    _eigvals(a)

Eigenvalues only. Registering just the one output is what tells the backend task
to skip the eigenvector computation.
"""
function _eigvals(a::NDArray)
    ew = _alloc_eigenvalues(a)
    geev!(a, ew, nothing)
    return ew
end

function _alloc_eigenvalues(a::NDArray{T,N}) where {T,N}
    _check_square_matrices(:eigen, a)
    _assert_geev_available()
    return cuNumeric.zeros(_eig_complex_eltype(T), size(a)[1:(end - 1)]...)
end

function _assert_geev_available()
    (Legate.num_gpus() > 0 && !cuNumeric.cusolver_has_geev()) && error(
        "eigen requires cusolverDnXgeev, which the installed cuSolver does not " *
        "provide. Upgrade CUDA (12.6.2 or newer) or run without GPUs.",
    )
    return nothing
end
