# Copyright 2024 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
# Task construction follows cuPyNumeric 26.06's linalg/_solve.py, _qr.py,
# and _cholesky.py.
# Keep algorithm selection here; Legate owns placement and redistribution.
const MIN_SOLVE_MATRIX_SIZE = 2048
const MIN_SOLVE_TILE_SIZE = 512
const MIN_CHOLESKY_MATRIX_SIZE = 8192
const MIN_CHOLESKY_TILE_SIZE = 2048
const MIN_QR_MATRIX_SIZE = 1048576
const QR_TILE_SIZE = 128
const MAX_CHOLESKY_TILES_PER_PROC = 4

struct _SingleProcLinalg end
struct _CuSolverMpLinalg end
struct _TiledCholesky end

# Neither the loaded library's capability nor the active machine is queried at
# module/precompile time. Explicit arguments also let policy tests run on CPUs.
function _linalg_backend(op, a::NDArray; kwargs...)
    return _linalg_backend(
        op, size(a), cusolvermp_available(), Int(Legate.num_gpus()), Int(Legate.num_procs());
        kwargs...,
    )
end

_mp_eligible(available::Bool, gpus::Integer) = available && gpus > 1

# Tuple length carries dimensionality in its type. Stacked systems never enter
# the MP selector; only their leading batch axes may be distributed.
_linalg_backend(::Val{:solve}, ::Tuple, available::Bool, gpus, procs) = _SingleProcLinalg()

function _linalg_backend(::Val{:solve}, shape::NTuple{2,Int}, available::Bool, gpus, procs)
    use_mp = shape[1] >= MIN_SOLVE_MATRIX_SIZE && _mp_eligible(available, gpus)
    return use_mp ? _CuSolverMpLinalg() : _SingleProcLinalg()
end

function _linalg_backend(::Val{:qr}, shape::Tuple{Int,Int}, available::Bool, gpus, procs)
    use_mp = (
        !iszero(min(shape...)) && prod(shape) >= MIN_QR_MATRIX_SIZE &&
        _mp_eligible(available, gpus)
    )
    return use_mp ? _CuSolverMpLinalg() : _SingleProcLinalg()
end

function _linalg_backend(
    ::Val{:cholesky}, ::Tuple, available::Bool, gpus, procs;
    lower::Bool=true, inplace::Bool=false,
)
    return _SingleProcLinalg()
end

function _linalg_backend(
    ::Val{:cholesky}, shape::NTuple{2,Int}, available::Bool, gpus, procs;
    lower::Bool=true, inplace::Bool=false,
)
    (!lower || inplace || procs == 1) && return _SingleProcLinalg()
    use_mp = shape[1] >= MIN_CHOLESKY_MATRIX_SIZE && _mp_eligible(available, gpus)
    return use_mp ? _CuSolverMpLinalg() : _TiledCholesky()
end

function _linalg_scalars!(task, args...)
    for arg in args
        Legate.add_scalar(task, Legate.Scalar(arg))
    end
    return nothing
end

function _linalg_manual_task(id, lo::Tuple{Int,Int}, hi::Tuple{Int,Int}; throws=false)
    task = create_linalg_task(id, Int64(lo[1]), Int64(lo[2]), Int64(hi[1]), Int64(hi[2]))
    task_throws_exception(task, throws)
    return task
end

_submit_linalg_task(task) = Legate.submit_manual_task(Legate.get_runtime(), task)

# Partition along rows, just as Python does. Every rank uses identical color
# spaces even when a reduced QR output has fewer rows than the input.
function _mp_row_partition(n::Int, gpus::Integer)
    n > 0 && gpus > 1 || throw(ArgumentError("MP tasks need nonempty inputs and multiple GPUs"))
    rows = cld(n, gpus)
    return rows, (cld(n, rows), 1)
end

function _check_mp_launch(tile::Integer)
    tile > 0 || throw(ArgumentError("cuSolverMp tile size must be positive"))
    cusolvermp_available() && Legate.num_gpus() > 1 ||
        throw(ArgumentError("cuSolverMp requires a supporting library and multiple active GPUs"))
    return nothing
end

_solve!(::_SingleProcLinalg, x, a, b) = solve_batched(a, b, x)

function _solve!(::_CuSolverMpLinalg, x, a, b; tile::Int=MIN_SOLVE_TILE_SIZE)
    _check_mp_launch(tile)
    n, nrhs = size(a, 1), size(b, 2)
    rows, colors = _mp_row_partition(n, Int(Legate.num_gpus()))
    pa = Legate.partition_by_tiling(nda_to_logical_store(a), (rows, n))
    pb = Legate.partition_by_tiling(nda_to_logical_store(b), (rows, nrhs))
    px = Legate.partition_by_tiling(nda_to_logical_store(x), (rows, nrhs))
    @task_scope "mp_solve" begin
        task = _linalg_manual_task(MP_SOLVE, (0, 0), (colors[1] - 1, 0); throws=true)
        Legate.add_input(task, pa)
        Legate.add_input(task, pb)
        Legate.add_output(task, px)
        _linalg_scalars!(task, Int64(n), Int64(nrhs), Int64(tile))
        add_nccl_communicator(task)
        _submit_linalg_task(task)
    end
    return x
end

function _qr(::_CuSolverMpLinalg, a::NDArray{T,2}; tile::Int=QR_TILE_SIZE) where {T}
    _check_mp_launch(tile)
    m, n = size(a)
    k = min(m, n)
    q, r = cuNumeric.zeros(T, m, k), cuNumeric.zeros(T, k, n)
    rows, colors = _mp_row_partition(m, Int(Legate.num_gpus()))
    tiles = (rows, n)
    pa = Legate.partition_by_tiling(nda_to_logical_store(a), tiles)
    pq = Legate.partition_by_tiling(nda_to_logical_store(q), tiles, colors)
    pr = Legate.partition_by_tiling(nda_to_logical_store(r), tiles, colors)
    @task_scope "mp_qr" begin
        task = _linalg_manual_task(MP_QR, (0, 0), (colors[1] - 1, 0); throws=true)
        Legate.add_input(task, pa)
        Legate.add_output(task, pq)
        Legate.add_output(task, pr)
        _linalg_scalars!(task, Int64(m), Int64(n), Int64(tile), Int64(tile))
        add_nccl_communicator(task)
        _submit_linalg_task(task)
    end
    return q, r
end

_cholesky!(::_SingleProcLinalg, out, a) = potrf!(out, a; lower=true, zeroout=true)

function _cholesky!(::_CuSolverMpLinalg, out, a; tile::Int=MIN_CHOLESKY_TILE_SIZE)
    _check_mp_launch(tile)
    @task_scope "mp_potrf" begin
        rt = Legate.get_runtime()
        task = Legate.create_auto_task(rt, get_lib(), MP_POTRF)
        task_throws_exception(task, true)
        ai = Legate.add_input(task, nda_to_logical_array(a))
        oi = Legate.add_output(task, nda_to_logical_array(out))
        Legate.add_constraint(task, Legate.align(oi, ai))
        _linalg_scalars!(task, Int64(size(a, 1)), Int64(tile))
        add_nccl_communicator(task)
        Legate.submit_auto_task(rt, task)
        _cholesky_tril!(out)
    end
    return out
end

function _cholesky_tril!(out::NDArray)
    rt = Legate.get_runtime()
    task = Legate.create_auto_task(rt, get_lib(), TRILU)
    store = nda_to_logical_array(out)
    Legate.add_output(task, store)
    Legate.add_input(task, store)
    # The third argument identifies Cholesky to the backend/mapper.
    _linalg_scalars!(task, true, Int32(0), true)
    Legate.submit_auto_task(rt, task)
    return nothing
end

function _cholesky_color_shape(
    n::Int, procs::Integer;
    min_matrix::Int=MIN_CHOLESKY_MATRIX_SIZE, min_tile::Int=MIN_CHOLESKY_TILE_SIZE,
)
    n > 0 && procs > 0 && min_matrix >= 0 && min_tile > 0 ||
        throw(ArgumentError("invalid tiled Cholesky dimensions or tile policy"))
    (procs == 1 || n <= min_matrix) && return (1, 1)
    tiles = Int(procs)
    while cld(n, tiles) > min_tile && 2 * tiles <= procs * MAX_CHOLESKY_TILES_PER_PROC
        tiles *= 2
    end
    return (tiles, tiles)
end

# Each task reads only tiles ready at this stage; Legate records the DAG from
# these inputs/outputs. No execution fence or host copy is needed between steps.
function _cholesky!(
    ::_TiledCholesky, out, a;
    min_matrix::Int=MIN_CHOLESKY_MATRIX_SIZE, min_tile::Int=MIN_CHOLESKY_TILE_SIZE,
)
    n = size(a, 1)
    initial = _cholesky_color_shape(n, Int(Legate.num_procs()); min_matrix, min_tile)
    tile = cld(n, initial[1])
    colors = cld(n, tile)
    pa = Legate.partition_by_tiling(nda_to_logical_store(a), (tile, tile))
    po = Legate.partition_by_tiling(nda_to_logical_store(out), (tile, tile))
    @task_scope "tiled_cholesky" begin
        task = _linalg_manual_task(TRANSPOSE_COPY_2D, (0, 0), (colors - 1, colors - 1))
        Legate.add_output(task, po)
        Legate.add_input(task, pa)
        _submit_linalg_task(task)
        for i in 0:(colors - 1)
            _cholesky_potrf!(po, i)
            _cholesky_trsm!(po, i, colors)
            for k in (i + 1):(colors - 1)
                _cholesky_syrk!(po, k, i)
                _cholesky_gemm!(po, k, i, colors)
            end
        end
        task = _linalg_manual_task(TRILU, (0, 0), (colors - 1, colors - 1))
        Legate.add_output(task, po)
        Legate.add_input(task, po)
        _linalg_scalars!(task, true, Int32(0), true)
        _submit_linalg_task(task)
    end
    return out
end

function _cholesky_potrf!(p, i)
    task = _linalg_manual_task(POTRF, (i, i), (i, i); throws=true)
    Legate.add_output(task, p)
    Legate.add_input(task, p)
    _linalg_scalars!(task, true, false)
    return _submit_linalg_task(task)
end

function _cholesky_trsm!(p, i, colors)
    i + 1 >= colors && return nothing
    task = _linalg_manual_task(TRSM, (i + 1, i), (colors - 1, i); throws=true)
    Legate.add_output(task, p)
    add_input_tile(task, p.handle, UInt64(i), UInt64(i))
    Legate.add_input(task, p)
    # Right-side solve with the conjugate transpose of the lower factor.
    _linalg_scalars!(task, false, true, Int32(2), false)
    return _submit_linalg_task(task)
end

function _cholesky_syrk!(p, k, i)
    task = _linalg_manual_task(SYRK, (k, k), (k, k))
    Legate.add_output(task, p)
    add_input_tile(task, p.handle, UInt64(k), UInt64(i))
    Legate.add_input(task, p)
    return _submit_linalg_task(task)
end

function _cholesky_gemm!(p, k, i, colors)
    k + 1 >= colors && return nothing
    task = _linalg_manual_task(GEMM, (k + 1, k), (colors - 1, k))
    Legate.add_output(task, p)
    add_input_column(task, p.handle, Int32(i))
    add_input_tile(task, p.handle, UInt64(k), UInt64(i))
    Legate.add_input(task, p)
    return _submit_linalg_task(task)
end
