const _MR_THREADS = 256
const _MR_PTX_CACHE = Dict{Any,String}()
const _MR_PTX_LOCK = ReentrantLock()

struct MapReduceMap{F,OP,R,N}
    f::F
    op::OP
    # Axes are runtime data; changing dims does not change the mapper/kernel type.
    mask::NTuple{N,Bool}
end
MapReduceMap(f::F, op::OP, ::Type{R}, mask::NTuple{N,Bool}) where {F,OP,R,N} =
    MapReduceMap{F,OP,R,N}(f, op, mask)

# All 32 lanes execute each shuffle, including lanes without input. Only valid
# sources enter the arithmetic: padding with an identity changes signed zeros
# and can turn complex infinities into NaNs.
@inline function _mr_reduce_warp(op, value, lane, active)
    offset = 16
    while offset > 0
        other = CUDACore.shfl_down_sync(0xffffffff, value, offset)
        if lane + offset <= active
            value = op(value, other)
        end
        offset >>= 1
    end
    return value
end

# The result is defined on thread 1. Both callers launch exactly _MR_THREADS
# threads and supply a contiguous prefix of valid thread-local accumulators.
@inline function _mr_reduce_block(op, value::S, active) where {S}
    tid = Int(CUDACore.threadIdx().x)
    lane = ((tid - 1) & 31) + 1
    warp = ((tid - 1) >> 5) + 1
    value = _mr_reduce_warp(op, value, lane, min(32, active - (warp - 1) * 32))
    shared = CUDACore.CuStaticSharedArray(S, _MR_THREADS ÷ 32)
    lane == 1 && (@inbounds shared[warp] = value)
    CUDACore.sync_threads()
    if warp == 1
        warps = (active + 31) >> 5
        if lane <= warps
            @inbounds value = shared[lane]
        end
        value = _mr_reduce_warp(op, value, lane, warps)
    end
    return value
end

# In one dimension the selected index is already bounded by the only extent.
# Preserve the physical stride without computing a remainder for every element.
@inline _mr_offset(A::CuStridedDeviceArray{T,1}, other::Int, red::Int, mask::NTuple{1,Bool}) where {T} =
    (mask[1] ? red : other) * A.strides[1]

# Indices are relative to the PhysicalStore's lower bound, already reflected in
# the descriptor pointer. Unchecked unsigned division avoids device exceptions.
@inline function _mr_offset(A::CuStridedDeviceArray{T,N}, other::Int, red::Int, mask::NTuple{N,Bool}) where {T,N}
    o, r = _bitcast_uint(other), _bitcast_uint(red)
    offset = 0
    @inbounds for d in 1:N
        extent = _bitcast_uint(A.dims[d])
        if mask[d]
            offset += _bitcast_int(Core.Intrinsics.urem_int(r, extent)) * A.strides[d]
            r = Core.Intrinsics.udiv_int(r, extent)
        else
            offset += _bitcast_int(Core.Intrinsics.urem_int(o, extent)) * A.strides[d]
            o = Core.Intrinsics.udiv_int(o, extent)
        end
    end
    return offset
end

function _mr_partial_kernel(A, scratch::CuStridedDeviceArray{S,1}, mapper::MapReduceMap{F,OP,R,N}, start::Int, chunks::Int) where {S,F,OP,R,N}
    tid = Int(CUDACore.threadIdx().x)
    block = Int(CUDACore.blockIdx().x) - 1
    other = start + _bitcast_int(Core.Intrinsics.udiv_int(_bitcast_uint(block), _bitcast_uint(chunks)))
    chunk = _bitcast_int(Core.Intrinsics.urem_int(_bitcast_uint(block), _bitcast_uint(chunks)))
    nred = 1
    @inbounds for d in 1:N
        mapper.mask[d] && (nred *= A.dims[d])
    end
    value = _mr_identity(mapper.op, S)
    i = chunk * _MR_THREADS + tid - 1
    first = true
    while i < nred
        offset = _mr_offset(A, other, i, mapper.mask)
        x = unsafe_load(pointer(A), offset + 1, Val(_strided_align(A)))
        mapped = _mr_encode(mapper.op, convert(R, mapper.f(x)))
        value = first ? mapped : _mr_combine(mapper.op)(value, mapped)
        first = false
        i += chunks * _MR_THREADS
    end
    active = min(_MR_THREADS, nred - chunk * _MR_THREADS)
    value = _mr_reduce_block(_mr_combine(mapper.op), value, active)
    tid == 1 && (@inbounds scratch[block + 1] = value)
    return nothing
end

# One thread owns each retained coordinate. The C++ launcher obtains this
# pointer from an exclusive Legate reduction accessor, as cuPyNumeric's GEMV
# task does. Legate, not this kernel, combines contributions between tasks.
function _mr_contribute_kernel(scratch, dest, op, single::Bool, start::Int, count::Int, chunks::Int)
    i = (Int(CUDACore.blockIdx().x) - 1) * Int(CUDACore.blockDim().x) + Int(CUDACore.threadIdx().x)
    if i <= count
        @inbounds value = scratch[(i - 1) * chunks + 1]
        for c in 2:chunks
            @inbounds value = _mr_combine(op)(value, scratch[(i - 1) * chunks + c])
        end
        @inbounds dest[start + i] = single ? value : _mr_combine(op)(dest[start + i], value)
    end
    return nothing
end

# A full reduction has one output. Cooperate across a block instead of making
# a single thread serially combine every partial. Invalid lanes never enter the
# tree: an extra identity operation can change signed zeros or complex infinities.
function _mr_contribute_full_kernel(scratch::CuStridedDeviceArray{S,1}, dest, op,
                                    single::Bool, start::Int, count::Int, chunks::Int) where {S}
    tid = Int(CUDACore.threadIdx().x)
    active = min(chunks, _MR_THREADS)
    value = _mr_identity(op, S)
    if tid <= active
        @inbounds value = scratch[tid]
        for c in (tid + _MR_THREADS):_MR_THREADS:chunks
            @inbounds value = _mr_combine(op)(value, scratch[c])
        end
    end
    value = _mr_reduce_block(_mr_combine(op), value, active)
    if tid == 1
        @inbounds dest[start + 1] = single ? value : _mr_combine(op)(dest[start + 1], value)
    end
    return nothing
end

struct MapReduceFinish{OP,R,I}
    op::OP
    init::I
end
MapReduceFinish(op::OP, ::Type{R}, init::I) where {OP,R,I} = MapReduceFinish{OP,R,I}(op, init)
@inline function (finish::MapReduceFinish{OP,R})(x) where {OP,R}
    return _mr_finish(_mr_combine(finish.op), _mr_decode(finish.op, R, x), finish.init)
end

struct MapReduceSingleton{F,OP,R,O,I}
    f::F
    op::OP
    init::I
end
function MapReduceSingleton(
    f::F, op::OP, ::Type{R}, ::Type{O}, init::I,
) where {F,OP,R,O,I}
    return MapReduceSingleton{F,OP,R,O,I}(f, op, init)
end
@inline function (finish::MapReduceSingleton{F,OP,R,O})(x) where {F,OP,R,O}
    mapped = convert(R, finish.f(x))
    value = _mr_finish(_mr_combine(finish.op), mapped, finish.init)
    return convert(O, value)
end
function _mr_finish_kernel(src, dest, finish)
    i = (Int(CUDACore.blockIdx().x) - 1) * Int(CUDACore.blockDim().x) + Int(CUDACore.threadIdx().x)
    step = Int(CUDACore.gridDim().x) * Int(CUDACore.blockDim().x)
    while i <= length(dest)
        @inbounds dest[i] = finish(src[i])
        i += step
    end
    return nothing
end

function _mr_kernel_name(kernel, types)
    target = (CUDACore.capability(CUDACore.device()), _COMPATIBLE_PTX_VERSION[])
    key = (kernel, types, target)
    return lock(_MR_PTX_LOCK) do
        get!(_MR_PTX_CACHE, key) do
            buf = IOBuffer()
            _emit_compatible_ptx(buf, kernel, types)
            ptx = String(take!(buf))
            original = extract_kernel_name(ptx)
            name = original * "_mr_" * string(hash(ptx); base=16)
            ptx_task(replace(ptx, original => name), name)
            name
        end
    end
end

_mr_dim_seed(op::_MR_OP, ::Type{R}, init, dims) where {R} = init
_mr_dim_seed(op::_MR_ADD, ::Type{R}, ::NoReductionInit, dims::_MR_DIMS) where {R} = zero(R)
_mr_dim_seed(op::_MR_MUL, ::Type{R}, ::NoReductionInit, dims::_MR_DIMS) where {R} = one(R)

function _mr_finish_name(finish::F, ::Type{T}, ::Type{O}, ::Val{D}) where {F,T,O,D}
    return _mr_kernel_name(_mr_finish_kernel, (
        CuStridedDeviceArray{T,D,CUDACore.AS.Global},
        CuStridedDeviceArray{O,D,CUDACore.AS.Global}, F,
    ))
end

function _mr_submit(A, accumulator, result, mask, full, single, redop,
                    name, contribute_name, finish_name, mapper, finish)
    axis_bits = sum(d -> UInt64(mask[d]) << (d - 1), 1:length(mask))
    mapper_ref, finish_ref = Ref(mapper), Ref(finish)
    @task_scope "mapreduce" begin
        # Submission copies these bytes into task-owned scalars. Borrowing
        # Refs avoids Julia-owned CxxWrap vector handles on every call.
        GC.@preserve A accumulator result mapper_ref finish_ref begin
            submit_mapreduce(
                CxxWrap.CxxPtr{CN_NDArray}(A.ptr),
                CxxWrap.CxxPtr{CN_NDArray}(accumulator.ptr),
                CxxWrap.CxxPtr{CN_NDArray}(result.ptr),
                axis_bits, full, single, redop, name, contribute_name, finish_name,
                Base.unsafe_convert(Ptr{Cvoid}, mapper_ref), sizeof(mapper),
                Base.unsafe_convert(Ptr{Cvoid}, finish_ref), sizeof(finish),
            )
        end
    end
    return result
end

function _mr_launch(f, op, A::NDArray{T,N}, ::Type{R}, ::Type{O}, mask, shape, init, dims, single::Bool) where {T,N,R,O}
    S = _mr_storage(op, R)
    D = max(N, 1)
    OD = max(length(shape), 1)
    seed = _mr_dim_seed(op, R, init, dims)
    if single && !(dims isa Colon)
        finish = MapReduceSingleton(f, op, R, O, seed)
        finish_name = _mr_finish_name(finish, T, O, Val(OD))
        result = nda_empty_array(shape, O)
        try
            # The singleton submission never uses the accumulator or mapper.
            return _mr_submit(A, result, result, mask, false, true, _mr_redop(op, S),
                              "", "", finish_name, nothing, finish)
        catch
            destroy!(result)
            rethrow()
        end
    end
    input_type = CuStridedDeviceArray{T,D,CUDACore.AS.Global}
    scratch_type = CuStridedDeviceArray{S,1,CUDACore.AS.Global}
    mapper = MapReduceMap(f, op, R, mask)
    name = _mr_kernel_name(_mr_partial_kernel, (input_type, scratch_type, typeof(mapper), Int, Int))
    RD = dims isa Colon ? 1 : D
    contribute_kernel = dims isa Colon ? _mr_contribute_full_kernel : _mr_contribute_kernel
    contribute_name = _mr_kernel_name(contribute_kernel, (
        scratch_type, CuStridedDeviceArray{S,RD,CUDACore.AS.Global}, typeof(op), Bool, Int, Int, Int,
    ))
    finish = MapReduceFinish(op, R, seed)
    needs_finish = S !== O || !(finish.init isa NoReductionInit)
    finish_name = needs_finish ? _mr_finish_name(finish, S, O, Val(OD)) : ""
    accumulator = single ? nda_empty_array(shape, S) : nda_full_array(shape, _mr_identity(op, S))
    result = nothing
    try
        result = needs_finish ? nda_empty_array(shape, O) : accumulator
        _mr_submit(A, accumulator, result, mask, dims isa Colon, single, _mr_redop(op, S),
                   name, contribute_name, finish_name, mapper, finish)
    catch
        isnothing(result) || destroy!(result)
        rethrow()
    finally
        result === accumulator || destroy!(accumulator)
    end
    return result
end
