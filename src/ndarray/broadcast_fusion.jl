
struct RuntimeBroadcastArg{J} end
struct StaticBroadcastArg{J} end

Base.@propagate_inbounds @inline function _gpu_broadcast_getindex(x, I)
    return @inbounds Base.Broadcast._broadcast_getindex(x, I)
end

Base.@propagate_inbounds @inline function _gpu_broadcast_getindex(x::Number, I)
    return x
end

Base.@propagate_inbounds @inline function _materialize_broadcast_arg(
    ::RuntimeBroadcastArg{J},
    runtime_args,
    static_args,
    I,
) where {J}
    arg = getfield(runtime_args, J)
    return @inbounds _gpu_broadcast_getindex(arg, I)
end

Base.@propagate_inbounds @inline function _materialize_broadcast_arg(
    ::StaticBroadcastArg{J},
    runtime_args,
    static_args,
    I,
) where {J}
    return getfield(static_args, J)
end

Base.@propagate_inbounds @inline function _materialize_broadcast_args(
    ::Tuple{},
    runtime_args,
    static_args,
    I,
)
    return ()
end

Base.@propagate_inbounds @inline function _materialize_broadcast_args(
    plan::Tuple,
    runtime_args,
    static_args,
    I,
)
    head = getfield(plan, 1)
    tail = Base.tail(plan)

    return (
        @inbounds(_materialize_broadcast_arg(head, runtime_args, static_args, I)),
        @inbounds(_materialize_broadcast_args(tail, runtime_args, static_args, I))...,
    )
end

_is_runtime_broadcast_arg(x::NDArray) = true
_is_runtime_broadcast_arg(x::Base.Broadcast.Extruded) = true
_is_runtime_broadcast_arg(x::Number) = true
_is_runtime_broadcast_arg(x) = false

function _push_runtime_arg!(runtime_args, arg_plan, x)
    push!(runtime_args, x)
    push!(arg_plan, RuntimeBroadcastArg{length(runtime_args)}())
    return nothing
end

function _push_static_arg!(static_args, arg_plan, x)
    isbits(x) || throw(
        ArgumentError(
            "Broadcast fusion cannot statically capture non-isbits broadcast leaf " *
            "$(repr(x)) of type $(typeof(x))",
        ),
    )

    push!(static_args, x)
    push!(arg_plan, StaticBroadcastArg{length(static_args)}())
    return nothing
end

function split_broadcast_args_for_kernel(args::Tuple)
    runtime_args = Any[]
    static_args = Any[]
    arg_plan = Any[]

    for arg in args
        if arg isa Base.RefValue
            value = arg[]

            # Keep ordinary numeric broadcast scalars dynamic so scalar values
            # do not cause recompilation.
            if value isa Number
                _push_runtime_arg!(runtime_args, arg_plan, value)
            else
                # Function singletons, Val{N}(), etc.
                _push_static_arg!(static_args, arg_plan, value)
            end

        elseif _is_runtime_broadcast_arg(arg)
            _push_runtime_arg!(runtime_args, arg_plan, arg)

        elseif isbits(arg)
            # Conservative fallback for singleton/isbits scalar broadcast leaves.
            _push_static_arg!(static_args, arg_plan, arg)

        else
            throw(
                ArgumentError(
                    "Broadcast fusion does not know how to lower argument " *
                    "$(repr(arg)) of type $(typeof(arg))",
                ),
            )
        end
    end

    return tuple(runtime_args...), tuple(static_args...), tuple(arg_plan...)
end

##############

# The indexing in these kernels is based off this internal function from julia/broadcast.jl

# Base.@propagate_inbounds function _broadcast_getindex(bc::Broadcasted{<:Any,<:Any,<:Any,<:Any}, I)
#     args = _getindex(bc.args, I)
#     return _broadcast_getindex_evalf(bc.f, args...)
# end

function make_linear_kernel(dest, bc::Base.Broadcast.Broadcasted, arg_plan, static_args)
    f = bc.f

    @kernel function broadcast_kernel_linear_splat(dest, runtime_args...)
        I = @index(Global, Linear)
        @inbounds args_modified = _materialize_broadcast_args(
            arg_plan, runtime_args, static_args, I
        )
        @inbounds dest[I] = Base.Broadcast._broadcast_getindex_evalf(f, args_modified...)
    end

    return broadcast_kernel_linear_splat
end

function make_cartesian_kernel(dest, bc::Base.Broadcast.Broadcasted, arg_plan, static_args)
    f = bc.f

    @kernel function broadcast_kernel_cartesian_splat(dest, runtime_args...)
        I = @index(Global, Cartesian)
        @inbounds args_modified = _materialize_broadcast_args(
            arg_plan, runtime_args, static_args, I
        )
        @inbounds dest[I] = Base.Broadcast._broadcast_getindex_evalf(f, args_modified...)
    end

    return broadcast_kernel_cartesian_splat
end

struct FusedBroadcastMetadata
    ctx::Any # KA.CompilerMetadata
    threads::Int
    blocks::Int
    cuda_task::CUDATask
end

const _BCAST_PTX_CACHE = Dict{Tuple{Any,DataType,DataType,Any},FusedBroadcastMetadata}()
const _BCAST_PTX_CACHE_LOCK = ReentrantLock()

cudevice_array_offset(::Type{T}) where {T<:CUDACore.CuDeviceArray} = 0
cudevice_array_offset(::Type{T}) where {T<:Base.Broadcast.Extruded} = Int(fieldoffset(T, 1))

stores_cudevicearray(::Type{T}) where {T<:CUDACore.CuDeviceArray} = true
stores_cudevicearray(::Type{T}) where {T<:Base.Broadcast.Extruded} = true
stores_cudevicearray(::Type{T}) where {T<:Number} = false
stores_cudevicearray(::Type{T}) where {T<:Bool} = false

function stores_cudevicearray(::Type{T}) where {T}
    throw(error("Broadcast fusion. Don't know what to do with type: $T"))
end

function find_cudevicearray_offsets_and_indices(::Type{BC_ARGS}) where {BC_ARGS}
    offsets = Vector{Int}()
    indices = Vector{Int}()
    for (i, T) in enumerate(fieldtypes(BC_ARGS))
        if stores_cudevicearray(T)
            offset = fieldoffset(BC_ARGS, i) + cudevice_array_offset(T)
            push!(offsets, offset)
            push!(indices, i)
        end
    end
    return tuple(offsets...), tuple(indices...)
end

function find_scalar_offsets_and_indices(::Type{BC_ARGS}) where {BC_ARGS}
    offsets = Vector{Int}()
    indices = Vector{Int}()
    for (i, T) in enumerate(fieldtypes(BC_ARGS))
        if (T <: Number)
            offset = fieldoffset(BC_ARGS, i)
            push!(offsets, offset)
            push!(indices, i)
        end
    end
    return tuple(offsets...), tuple(indices...)
end

get_ndarray(x::T) where {T<:NDArray} = x
get_ndarray(x::T) where {T<:Base.Broadcast.Extruded} = x.x
get_ndarray(x::T) where {T} = throw(error("Broadcast fusion. Don't know what to do with type: $T"))

function _threads_from_occupancy(
    obj::KA.Kernel{CUDACore.CUDAKernels.CUDABackend},
    ::Type{DEST_T},
    ARG_TYPES...;
    ndrange,
) where {DEST_T}
    backend = KA.backend(obj)

    ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(obj, ndrange, nothing)
    ctx = KA.mkcontext(obj, ndrange, iterspace)

    # If the kernel is statically sized we can tell the compiler about that
    maxthreads =
        if KA.workgroupsize(obj) <: KA.StaticSize
            prod(KA.get(KA.workgroupsize(obj)))
        else
            nothing
        end

    # Determine threads via occupancy
    tt = Base.to_tuple_type((typeof(ctx), DEST_T, ARG_TYPES...))
    host_kernel = CUDACore.cufunction(
        obj.f,
        tt;
        kernel=true,
        maxthreads=maxthreads,
        always_inline=backend.always_inline,
    )
    config = CUDACore.launch_configuration(host_kernel.fun; max_threads=prod(ndrange))
    threads = config.threads

    workgroupsize = CUDACore.CUDAKernels.threads_to_workgroupsize(threads, ndrange)
    iterspace, dynamic = KA.partition(obj, ndrange, workgroupsize)
    ctx = KA.mkcontext(obj, ndrange, iterspace)

    blocks = length(KA.blocks(iterspace))
    threads = length(KA.workitems(iterspace))

    return threads, blocks, ctx
end

"""
    get_ptx(obj::KA.Kernel{CUDABackend}, ::Type{DEST_T}, ::Type{BC_T};
                         ndrange) -> (ptx::String, threads::Int, blocks::Int)

Compile a KA CUDA kernel (kernel-body `obj.f(ctx, ...)`) using *types only* for `DEST_T` and `BC_T`,
choose a workgroup size (threads) using CUDA occupancy when possible, and return the generated PTX.
"""
function get_ptx(
    obj::KA.Kernel{CUDACore.CUDAKernels.CUDABackend},
    ::Type{DEST_T},
    arg_types...;
    ndrange,
) where {DEST_T,BC_T}
    # println(Base.isbitstype.(arg_types))
    threads, blocks, ctx = _threads_from_occupancy(obj, DEST_T, arg_types...; ndrange=ndrange)
    blocks == 0 && return "", 0, 0, ctx #! MAYBE ERROR HERE?

    # Generate PTX
    buf = IOBuffer()
    #!TODO REMOVE THE MANUAL PTX VERSION HERE!
    CUDATools.code_ptx(buf, obj.f, (typeof(ctx), DEST_T, arg_types...);
        raw=false, kernel=true, ptx=v"7.8")

    return String(take!(buf)), threads, blocks, ctx
end

function get_cuda_task(
    obj::KA.Kernel{CUDACore.CUDAKernels.CUDABackend},
    dest::D,
    runtime_args::RT,
    ndrange,
) where {D<:NDArray,RT<:Tuple}
    DEST_T = map_cuda_type(D)
    ARG_TYPES = map_cuda_type.(typeof.(runtime_args))

    key = (obj, D, RT, ndrange)

    lock(_BCAST_PTX_CACHE_LOCK) do
        return get!(_BCAST_PTX_CACHE, key) do
            ptx, threads, blocks, ctx = get_ptx(obj, DEST_T, ARG_TYPES...; ndrange=ndrange)

            orig_name = extract_kernel_name(ptx)
            unique_name = orig_name * "_" * string(hash(ptx); base=16)
            ptx = replace(ptx, orig_name => unique_name)

            ptx_task(ptx, unique_name)
            cuda_task = CUDATask(unique_name, (DEST_T, ARG_TYPES...))

            FusedBroadcastMetadata(ctx, threads, blocks, cuda_task)
        end
    end
end

function fuse_broadcast_tree!(dest::D, bc::B) where {D<:NDArray,B<:Base.Broadcast.Broadcasted}
    bc = Base.Broadcast.preprocess(dest, bc)
    bc = Base.Broadcast.instantiate(bc)
    bc = Base.Broadcast.flatten(bc)

    # Things like exponentiation generate arguments like Base.RefValue
    # which do not work with our pattern for making CUDA kernels as they are
    # not is-bits types. We split these out manually into static args and handle
    # them separately from runtime args (i.e., arrays, scalars)
    runtime_args, static_args, arg_plan = split_broadcast_args_for_kernel(bc.args)

    broadcast_kernel =
        if ndims(dest) == 1 ||
            (isa(IndexStyle(dest), IndexLinear) &&
            isa(IndexStyle(bc), IndexLinear))
            make_linear_kernel(dest, bc, arg_plan, static_args)
        else
            make_cartesian_kernel(dest, bc, arg_plan, static_args)
        end

    ndrange = ndims(dest) > 0 ? size(dest) : (1,)

    bck_cuda = broadcast_kernel(CUDACore.CUDAKernels.CUDABackend())

    fkm = get_cuda_task(bck_cuda, dest, runtime_args, ndrange)

    num_outputs = 1

    unique_ndarrays = NDArray[]
    ndarray_to_input_idx = Dict{UInt,Int}()

    arg_map = Int32[]
    actual_scalars = Any[]

    # First PTX argument after ctx is dest.
    push!(arg_map, Int32(0))

    # Now map only runtime args, not original bc.args.
    for arg in runtime_args
        if stores_cudevicearray(map_cuda_type(typeof(arg)))
            nda = get_ndarray(arg)
            oid = objectid(nda)

            if !haskey(ndarray_to_input_idx, oid)
                push!(unique_ndarrays, nda)
                ndarray_to_input_idx[oid] = length(unique_ndarrays) - 1
            end

            input_idx = ndarray_to_input_idx[oid]
            push!(arg_map, Int32(num_outputs + input_idx))
        else
            push!(arg_map, Int32(-1 - length(actual_scalars)))
            push!(actual_scalars, arg)
        end
    end

    input_ndarrays = tuple(unique_ndarrays...)

    launch(
        fkm.cuda_task,
        input_ndarrays,
        (dest,),
        (Int32(length(arg_map)), arg_map..., actual_scalars...);
        blocks=fkm.blocks,
        threads=fkm.threads,
        taskid=cuNumeric.RUN_PTX_BROADCAST,
        ctx=fkm.ctx,
    )

    #! PROMOTION CHECKS?
    return dest
end
