const _MR_THREADS = 256
const _MR_PTX_CACHE = Dict{Any,String}()
const _MR_PTX_LOCK = ReentrantLock()

struct MapReduceMap{F,OP,R,MASK}
    f::F
    op::OP
end
MapReduceMap(f::F, op::OP, ::Type{R}, mask) where {F,OP,R} = MapReduceMap{F,OP,R,mask}(f, op)

# Indices are relative to the PhysicalStore's lower bound, already reflected in
# the descriptor pointer. Unchecked unsigned division avoids device exceptions.
@inline function _mr_offset(A::CuStridedDeviceArray{T,N}, other::Int, red::Int, ::Val{MASK}) where {T,N,MASK}
    o, r = _bitcast_uint(other), _bitcast_uint(red)
    offset = 0
    @inbounds for d in 1:N
        extent = _bitcast_uint(A.dims[d])
        if MASK[d]
            offset += _bitcast_int(Core.Intrinsics.urem_int(r, extent)) * A.strides[d]
            r = Core.Intrinsics.udiv_int(r, extent)
        else
            offset += _bitcast_int(Core.Intrinsics.urem_int(o, extent)) * A.strides[d]
            o = Core.Intrinsics.udiv_int(o, extent)
        end
    end
    return offset
end

function _mr_partial_kernel(A, scratch::CuStridedDeviceArray{S,1}, mapper::MapReduceMap{F,OP,R,MASK}, start::Int, chunks::Int) where {S,F,OP,R,MASK}
    tid = Int(CUDACore.threadIdx().x)
    block = Int(CUDACore.blockIdx().x) - 1
    other = start + _bitcast_int(Core.Intrinsics.udiv_int(_bitcast_uint(block), _bitcast_uint(chunks)))
    chunk = _bitcast_int(Core.Intrinsics.urem_int(_bitcast_uint(block), _bitcast_uint(chunks)))
    nred = 1
    @inbounds for d in 1:length(MASK)
        MASK[d] && (nred *= A.dims[d])
    end
    value = _mr_identity(mapper.op, S)
    i = chunk * _MR_THREADS + tid - 1
    first = true
    while i < nred
        offset = _mr_offset(A, other, i, Val(MASK))
        x = unsafe_load(pointer(A), offset + 1, Val(_strided_align(A)))
        mapped = _mr_encode(mapper.op, convert(R, mapper.f(x)))
        value = first ? mapped : _mr_combine(mapper.op)(value, mapped)
        first = false
        i += chunks * _MR_THREADS
    end
    shared = CUDACore.CuStaticSharedArray(S, _MR_THREADS)
    @inbounds shared[tid] = value
    stride = _MR_THREADS ÷ 2
    while stride > 0
        CUDACore.sync_threads()
        if tid <= stride && tid + stride <= min(_MR_THREADS, nred - chunk * _MR_THREADS)
            @inbounds shared[tid] = _mr_combine(mapper.op)(shared[tid], shared[tid + stride])
        end
        stride >>= 1
    end
    tid == 1 && (@inbounds scratch[block + 1] = shared[1])
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

function _mr_launch(f, op, A::NDArray{T,N}, ::Type{R}, ::Type{O}, mask, shape, init, dims) where {T,N,R,O}
    S = _mr_storage(op, R)
    D = max(N, 1)
    input_type = CuStridedDeviceArray{T,D,CUDACore.AS.Global}
    scratch_type = CuStridedDeviceArray{S,1,CUDACore.AS.Global}
    mapper = MapReduceMap(f, op, R, mask)
    single = all(d -> !mask[d] || size(A, d) == 1, 1:N)
    name = _mr_kernel_name(_mr_partial_kernel, (input_type, scratch_type, typeof(mapper), Int, Int))
    RD = dims isa Colon ? 1 : D
    contribute_name = _mr_kernel_name(_mr_contribute_kernel, (
        scratch_type, CuStridedDeviceArray{S,RD,CUDACore.AS.Global}, typeof(op), Bool, Int, Int, Int,
    ))
    singleton = single && !(dims isa Colon)
    finish = singleton ? MapReduceSingleton(f, op, R, O, _mr_dim_seed(op, R, init, dims)) :
                         MapReduceFinish(op, R, _mr_dim_seed(op, R, init, dims))
    OD = max(length(shape), 1)
    needs_finish = singleton || S !== O || !(finish.init isa NoReductionInit)
    finish_name = needs_finish ? _mr_kernel_name(_mr_finish_kernel, (
        CuStridedDeviceArray{singleton ? T : S,OD,CUDACore.AS.Global},
        CuStridedDeviceArray{O,OD,CUDACore.AS.Global}, typeof(finish),
    )) : ""
    accumulator = nda_full_array(shape, _mr_identity(op, S))
    result = nothing
    try
        result = needs_finish ? nda_zeros_array(shape, O) : accumulator
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
                    axis_bits, dims isa Colon, single, _mr_redop(op, S), name, contribute_name, finish_name,
                    Base.unsafe_convert(Ptr{Cvoid}, mapper_ref), sizeof(mapper),
                    Base.unsafe_convert(Ptr{Cvoid}, finish_ref), sizeof(finish),
                )
            end
        end
    catch
        isnothing(result) || destroy!(result)
        rethrow()
    finally
        result === accumulator || destroy!(accumulator)
    end
    return result
end
