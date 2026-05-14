
# The indexing in these kernels is based off this internal function from julia/broadcast.jl

# Base.@propagate_inbounds function _broadcast_getindex(bc::Broadcasted{<:Any,<:Any,<:Any,<:Any}, I)
#     args = _getindex(bc.args, I)
#     return _broadcast_getindex_evalf(bc.f, args...)
# end

function make_linear_kernel(dest, bc::Base.Broadcast.Broadcasted)
    f = bc.f

    @kernel function broadcast_kernel_linear_splat(dest, args...)
        I = @index(Global, Linear)
        @inbounds args_modified = Base.Broadcast._getindex(args, I)
        @inbounds dest[I] = Base.Broadcast._broadcast_getindex_evalf(f, args_modified...)
    end

    return broadcast_kernel_linear_splat
end

function make_cartesian_kernel(dest, bc::Base.Broadcast.Broadcasted)
    f = bc.f

    @kernel function broadcast_kernel_cartesian_splat(dest, args...)
        I = @index(Global, Cartesian)
        @inbounds args_modified = Base.Broadcast._getindex(args, I)
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
get_ndarray(x) = throw(error("Broadcast fusion. Don't know what to do with type: $T"))

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
    bc::B,
    ndrange,
) where {D<:NDArray,B<:Base.Broadcast.Broadcasted}
    DEST_T = map_cuda_type(D)
    ARG_TYPES = map_cuda_type.(typeof.(bc.args))

    key = (obj, D, B, ndrange)
    lock(_BCAST_PTX_CACHE_LOCK) do
        # Also stores in cache if not found in Dict
        return get!(_BCAST_PTX_CACHE, key) do
            ptx, threads, blocks, ctx = get_ptx(obj, DEST_T, ARG_TYPES...; ndrange=ndrange)
            orig_name = extract_kernel_name(ptx)
            # Append a PTX content hash to make each name unique.
            unique_name = orig_name * "_" * string(hash(ptx); base=16)
            ptx = replace(ptx, orig_name => unique_name)
            # println(ptx)
            ptx_task(ptx, unique_name)
            cuda_task = CUDATask(unique_name, (DEST_T, ARG_TYPES...))
            FusedBroadcastMetadata(ctx, threads, blocks, cuda_task)
        end
    end
end

function fuse_broadcast_tree!(dest::D, bc::B) where {D<:NDArray,B<:Base.Broadcast.Broadcasted}

    #! HOW DOES THIS BEHAVE WHEN BC HAS 2 RESULT ARRAYS?

    # Normalize the Braodcasted type
    bc = Base.Broadcast.preprocess(dest, bc)
    bc = Base.Broadcast.instantiate(bc)
    bc = Base.Broadcast.flatten(bc)

    # Get proper kernel
    broadcast_kernel =
        if ndims(dest) == 1 ||
            (isa(IndexStyle(dest), IndexLinear) &&
            isa(IndexStyle(bc), IndexLinear))
            make_linear_kernel(dest, bc)
        else
            make_cartesian_kernel(dest, bc)
        end

    ndrange = ndims(dest) > 0 ? size(dest) : (1,)

    # Tell KernelAsbtractions.jl we are using CUDA.jl
    bck_cuda = broadcast_kernel(CUDACore.CUDAKernels.CUDABackend())

    # Lookup in cache, if not found, compile and cache
    fkm = get_cuda_task(bck_cuda, dest, bc, ndrange)

    # Replace NDArrays with CuDeviceArrays in the Broadcasted type so we can figure out bit-offsets
    spoofed_bc_type = map_cuda_type(typeof(bc))
    fieldname(spoofed_bc_type, 3) == :args ||
        throw(ArgumentError("Broadcasted field 3 is not args. Failed to fuse broadcast."))

    args_offset = Int(fieldoffset(spoofed_bc_type, 3))
    args_offset == 0 ||
        throw(
            ArgumentError(
                "Broadcast fusion only supports Broadcasted layouts where args starts at offset 0; got offset $args_offset. This is likely a bug in the compiler."
            ),
        )

    # Spoof NDArrays with CuDeviceArrays in the Broadcasted type
    spoofed_bc_args_type = map_cuda_type(typeof(bc.args))

    # Build the arg_map: for each kernel arg (after ctx), record its source.
    # Convention: index into combined [outputs..., inputs...] for NDArrays.
    #   idx < num_outputs → output[idx]
    #   idx >= num_outputs → input[idx - num_outputs]
    # Scalar values are passed separately after the mapping.
    #
    # PTX kernel signature: f(kernel_state, ctx, dest, bc.args...)
    # So the arg_map covers: [dest, bc.args...] in order.

    num_outputs = 1  # dest is always the single output

    # Deduplicate NDArray inputs: map each unique NDArray to a unique input index.
    # The arg_map can reference the same input index multiple times (e.g. a .+ a).
    unique_ndarrays = NDArray[]
    ndarray_to_input_idx = Dict{UInt,Int}()  # objectid → input index

    arg_map = Int32[]
    actual_scalars = Any[]

    # First arg in PTX (after ctx) is always dest = output[0]
    push!(arg_map, Int32(0))  # output[0]

    # Then bc.args... in order
    for (i, arg) in enumerate(bc.args)
        if stores_cudevicearray(map_cuda_type(typeof(arg)))
            nda = get_ndarray(arg)
            oid = objectid(nda)
            if !haskey(ndarray_to_input_idx, oid)
                push!(unique_ndarrays, nda)
                ndarray_to_input_idx[oid] = length(unique_ndarrays) - 1  # 0-based
            end
            input_idx = ndarray_to_input_idx[oid]
            push!(arg_map, Int32(num_outputs + input_idx))  # offset by num_outputs
        else
            # Scalar — record its position and save the value
            push!(arg_map, Int32(-1 - length(actual_scalars)))  # negative = scalar
            push!(actual_scalars, arg)
        end
    end

    input_ndarrays = tuple(unique_ndarrays...)

    # @show arg_map actual_scalars length(unique_ndarrays)

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

    #! DO I NEED TO DO TYPE PROMOTION CHECKS??
    return dest
end

# STEP 1: Figure out bit-offsets for CuDeviceArrays and scalars in args of spoofed type.
# The spoofed type has the same fields and alignment that the PTX kernel expects.
# cudevicearray_offsets, cudevicearray_indices =
#     find_cudevicearray_offsets_and_indices(spoofed_bc_args_type)

# scalar_offsets, scalar_indices = find_scalar_offsets_and_indices(spoofed_bc_args_type)

# # STEP 2: Get NDarrays corresponding to the offsets in the spoofed type.
# input_ndarrays = ntuple(
#     i -> get_ndarray(bc.args[cudevicearray_indices[i]]), length(cudevicearray_indices)
# )
# input_scalars = ntuple(i -> bc.args[scalar_indices[i]], length(scalar_indices))
# patch_info = BroadcastPatchInfo(
#     sizeof(spoofed_bc_type),
#     input_ndarrays,
#     cudevicearray_offsets,
#     ntuple(i -> i - 1, length(input_ndarrays)),
#     scalar_offsets,
#     input_scalars,
# )
