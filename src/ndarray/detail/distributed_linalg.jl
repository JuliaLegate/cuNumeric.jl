# Copyright 2024 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
# Task construction follows cuPyNumeric 26.06's linalg/_solve.py, _qr.py,
# and _cholesky.py.
# Keep algorithm selection here; Legate owns placement and redistribution.
struct _LinalgRuntime
    available::Bool
    gpus::Int
    procs::Int
    mp_eligible::Bool
end

_LinalgRuntime(available::Bool, gpus::Int, procs::Int) =
    _LinalgRuntime(available, gpus, procs, available && gpus > 1)

# Populated once in _start_runtime(), including deferred initialization.
# These describe the configured machine for the lifetime of this runtime.
const _LINALG_RUNTIME = Ref(_LinalgRuntime(false, 0, 0))

struct _SingleProcLinalg end
struct _CuSolverMpLinalg end
struct _TiledCholesky end

const _LINALG_CONJ_TRANSPOSE = Int32(2)

_linalg_backend(op, a::NDArray) = _linalg_backend(op, size(a), _LINALG_RUNTIME[])

# Tuple length carries dimensionality in its type. Stacked systems never enter
# the MP selector; only their leading batch axes may be distributed.
_linalg_backend(::Val{:solve}, ::Tuple, ::_LinalgRuntime) = _SingleProcLinalg()

function _linalg_backend(::Val{:solve}, shape::NTuple{2,Int}, rt::_LinalgRuntime)
    use_mp = rt.mp_eligible && shape[1] >= MIN_SOLVE_MATRIX_SIZE
    return use_mp ? _CuSolverMpLinalg() : _SingleProcLinalg()
end

function _linalg_backend(::Val{:qr}, shape::NTuple{2,Int}, rt::_LinalgRuntime)
    use_mp = rt.mp_eligible && prod(shape) >= MIN_QR_MATRIX_SIZE
    return use_mp ? _CuSolverMpLinalg() : _SingleProcLinalg()
end

_linalg_backend(::Val{:cholesky}, ::Tuple, ::_LinalgRuntime) = _SingleProcLinalg()

function _linalg_backend(::Val{:cholesky}, shape::NTuple{2,Int}, rt::_LinalgRuntime)
    rt.procs == 1 && return _SingleProcLinalg()
    use_mp = rt.mp_eligible && shape[1] >= MIN_CHOLESKY_MATRIX_SIZE
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

# Partitions own their store; submitted tasks retain their own references.
# Drop Julia's temporary owners promptly instead of waiting for GC. Recursion
# keeps earlier partitions protected if creating a later partition fails.
_with_linalg_partitions(f) = f()
function _with_linalg_partitions(f, spec::Tuple, specs::Tuple...)
    store = nda_to_logical_store(first(spec))
    partition = try
        Legate.partition_by_tiling(store, Base.tail(spec)...)
    finally
        finalize(store.handle)
    end
    try
        return _with_linalg_partitions(specs...) do parts...
            f(partition, parts...)
        end
    finally
        finalize(partition.handle)
    end
end

# Partition along rows, just as Python does. Every rank uses identical color
# spaces even when a reduced QR output has fewer rows than the input.
function _mp_row_partition(n::Int, gpus::Integer)
    rows = cld(n, gpus)
    return rows, (cld(n, rows), 1)
end

_solve!(::_SingleProcLinalg, x, a, b) = solve_batched(a, b, x)

function _solve!(::_CuSolverMpLinalg, x, a, b)
    n, nrhs = size(a, 1), size(b, 2)
    rows, colors = _mp_row_partition(n, _LINALG_RUNTIME[].gpus)
    _with_linalg_partitions(
        (a, (rows, n)), (b, (rows, nrhs)), (x, (rows, nrhs))
    ) do pa, pb, px
        @task_scope "mp_solve" begin
            task = _linalg_manual_task(MP_SOLVE, (0, 0), (colors[1] - 1, 0); throws=true)
            Legate.add_input(task, pa)
            Legate.add_input(task, pb)
            Legate.add_output(task, px)
            _linalg_scalars!(task, Int64(n), Int64(nrhs), Int64(MIN_SOLVE_TILE_SIZE))
            add_nccl_communicator(task)
            _submit_linalg_task(task)
        end
    end
    return x
end

function _qr!(::_CuSolverMpLinalg, q, r, a)
    m, n = size(a)
    rows, colors = _mp_row_partition(m, _LINALG_RUNTIME[].gpus)
    tiles = (rows, n)
    _with_linalg_partitions((a, tiles), (q, tiles, colors), (r, tiles, colors)) do pa, pq, pr
        @task_scope "mp_qr" begin
            task = _linalg_manual_task(MP_QR, (0, 0), (colors[1] - 1, 0); throws=true)
            Legate.add_input(task, pa)
            Legate.add_output(task, pq)
            Legate.add_output(task, pr)
            _linalg_scalars!(task, Int64(m), Int64(n), Int64(QR_TILE_SIZE), Int64(QR_TILE_SIZE))
            add_nccl_communicator(task)
            _submit_linalg_task(task)
        end
    end
    return nothing
end

_cholesky!(::_SingleProcLinalg, out, a) = potrf!(out, a; lower=true, zeroout=true)

function _cholesky!(::_CuSolverMpLinalg, out, a)
    @task_scope "mp_potrf" begin
        rt = Legate.get_runtime()
        task = Legate.create_auto_task(rt, get_lib(), MP_POTRF)
        task_throws_exception(task, true)
        ai = _add_task_array!(Legate.add_input, task, a)
        oi = _add_task_array!(Legate.add_output, task, out)
        Legate.add_constraint(task, Legate.align(oi, ai))
        _linalg_scalars!(task, Int64(size(a, 1)), Int64(MIN_CHOLESKY_TILE_SIZE))
        add_nccl_communicator(task)
        Legate.submit_auto_task(rt, task)
        _cholesky_tril!(out)
    end
    return out
end

function _cholesky_tril!(out::NDArray)
    rt = Legate.get_runtime()
    task = Legate.create_auto_task(rt, get_lib(), TRILU)
    _add_task_array!(Legate.add_output, task, out)
    _add_task_array!(Legate.add_input, task, out)
    # The third argument identifies Cholesky to the backend/mapper.
    _linalg_scalars!(task, true, Int32(0), true)
    Legate.submit_auto_task(rt, task)
    return nothing
end

function _cholesky_color_shape(n::Int, procs::Int)
    (procs == 1 || n <= MIN_CHOLESKY_MATRIX_SIZE) && return (1, 1)
    tiles = Int(procs)
    while cld(n, tiles) > MIN_CHOLESKY_TILE_SIZE &&
        2 * tiles <= procs * MAX_CHOLESKY_TILES_PER_PROC
        tiles *= 2
    end
    return (tiles, tiles)
end

# Each task reads only tiles ready at this stage; Legate records the DAG from
# these inputs/outputs. No execution fence or host copy is needed between steps.
function _cholesky!(::_TiledCholesky, out, a)
    n = size(a, 1)
    initial = _cholesky_color_shape(n, _LINALG_RUNTIME[].procs)
    tile = cld(n, initial[1])
    colors = cld(n, tile)
    _with_linalg_partitions((a, (tile, tile)), (out, (tile, tile))) do pa, po
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
    _linalg_scalars!(task, false, true, _LINALG_CONJ_TRANSPOSE, false)
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
