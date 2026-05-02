
# Copied from GPUArrays: https://github.com/JuliaGPU/GPUArrays.jl/blob/a9df2ba41ca2358c1de2f3cc6b020578bf6e39b1/src/host/broadcast.jl#L60-L63
# Defined with KernelAbstractions.jl. Makes it easier to generate indexing for
# various dimensions of inputs/outputs. Assumes broadcast is `Base.Broadcast.process`ed so that
# dest/bc have singleton dimensions inserted and we can index 1-1 like this.
@kernel function broadcast_kernel_cartesian(dest, bc)
    I = @index(Global, Cartesian)
    @inbounds dest[I] = bc[I]
end

@kernel function broadcast_kernel_linear(dest, bc)
    I = @index(Global, Linear)
    @inbounds dest[I] = bc[I]
end

# No compilation here, just generating CUDA specific kernel.
const GPU_CARTESIAN_KERNEL = broadcast_kernel_cartesian(CUDACore.CUDAKernels.CUDABackend())
const GPU_LINEAR_KERNEL = broadcast_kernel_linear(CUDACore.CUDAKernels.CUDABackend())

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

"""
    get_ptx(obj::KA.Kernel{CUDABackend}, ::Type{DEST_T}, ::Type{BC_T};
                         ndrange) -> (ptx::String, threads::Int, blocks::Int)

Compile a KA CUDA kernel (kernel-body `obj.f(ctx, ...)`) using *types only* for `DEST_T` and `BC_T`,
choose a workgroup size (threads) using CUDA occupancy when possible, and return the generated PTX.
"""
function get_ptx(
    obj::KA.Kernel{CUDACore.CUDAKernels.CUDABackend},
    ::Type{DEST_T},
    ::Type{BC_T};
    ndrange,
) where {DEST_T,BC_T}
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

    # Determine threads via occupancy if we can compile from types; otherwise use a broadcast-friendly heuristic.
    threads = _default_broadcast_threads(ndrange)
    tt = Base.to_tuple_type((typeof(ctx), DEST_T, BC_T))
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
    blocks == 0 && return "", 0, 0, ctx

    buf = IOBuffer()
    CUDATools.code_ptx(buf, obj.f, (typeof(ctx), DEST_T, BC_T); raw=false, kernel=true, ptx=v"7.8")
    return String(take!(buf)), threads, blocks, ctx
end

function get_cuda_task(
    obj::KA.Kernel{CUDACore.CUDAKernels.CUDABackend},
    dest::D,
    bc::B,
    ndrange,
) where {D<:NDArray,B<:Base.Broadcast.Broadcasted}
    DEST_T = map_cuda_type(D)
    BC_T = map_cuda_type(B)

    key = (obj, D, B, ndrange)
    lock(_BCAST_PTX_CACHE_LOCK) do
        # Also stores in cache if not found in Dict
        return get!(_BCAST_PTX_CACHE, key) do
            ptx, threads, blocks, ctx = get_ptx(obj, DEST_T, BC_T; ndrange=ndrange)
            func_name = extract_kernel_name(ptx)
            # println(ptx)
            ptx_task(ptx, func_name)
            cuda_task = CUDATask(func_name, (DEST_T, BC_T))
            FusedBroadcastMetadata(ctx, threads, blocks, cuda_task)
        end
    end
end

function fuse_broadcast_tree!(dest::D, bc::B) where {D<:NDArray,B<:Base.Broadcast.Broadcasted}

    #! HOW DOES THIS BEHAVE WHEN BC HAS 2 RESULT ARRAYS?

    bc = Base.Broadcast.preprocess(dest, bc)
    bc = Base.Broadcast.instantiate(bc)
    bc = Base.Broadcast.flatten(bc)

    # Get proper kernel
    broadcast_kernel =
        if ndims(dest) == 1 ||
            (isa(IndexStyle(dest), IndexLinear) &&
            isa(IndexStyle(bc), IndexLinear))
            GPU_LINEAR_KERNEL
        else
            GPU_CARTESIAN_KERNEL
        end

    ndrange = ndims(dest) > 0 ? size(dest) : (1,)

    # Lookup in cache, if not found, compile and cache
    fused_kernel_metadata = get_cuda_task(broadcast_kernel, dest, bc, ndrange)

    # Replace NDArrays with CuDeviceArrays in the Broadcasted type so we can figure out bit-offsets
    spoofed_bc_type = map_cuda_type(typeof(bc))
    fieldname(spoofed_bc_type, 3) == :args ||
        throw(ArgumentError("Broadcasted field 3 is not args. Failed to fuse broadcast."))
    args_offset = Int(fieldoffset(spoofed_bc_type, 3))

    # Replace NDArrays with CuDeviceArrays in the Broadcasted type so we can figure out bit-offsets
    spoofed_bc_args_type = map_cuda_type(typeof(bc.args))

    #!TODO FIGURE OUT HOW TO HANDLE AXES
    #! TODO FIGURE OUT IF ITS SAFE TO IGNORE FIRST TWO FIELDS OF BROADCASTED TYPE

    # STEP 1: Figure out bit-offsets for CuDeviceArrays and scalars in args of spoofed type.
    # The spoofed type has the same fields and alignment that the PTX kernel expects.
    cudevicearray_offsets, cudevicearray_indices = find_cudevicearray_offsets_and_indices(
        spoofed_bc_args_type
    )
    scalar_offsets, scalar_indices = find_scalar_offsets_and_indices(spoofed_bc_args_type)
    cudevicearray_offsets = args_offset .+ cudevicearray_offsets
    scalar_offsets = args_offset .+ scalar_offsets
    # STEP 2: Get NDarrays corresponding to the offsets in the spoofed type.
    input_ndarrays = ntuple(
        i -> get_ndarray(bc.args[cudevicearray_indices[i]]), length(cudevicearray_indices)
    )
    input_scalars = ntuple(i -> bc.args[scalar_indices[i]], length(scalar_indices))
    patch_info = BroadcastPatchInfo(
        sizeof(spoofed_bc_type),
        input_ndarrays,
        cudevicearray_offsets,
        ntuple(i -> i - 1, length(input_ndarrays)),
        scalar_offsets,
        input_scalars,
    )

    println("Array Offsets: ", cudevicearray_offsets)
    println("Scalar Offsets: ", scalar_offsets)
    println("Input NDArrays: ", input_ndarrays)
    println("Input Scalars: ", input_scalars)

    # launch_broadcast(
    #     fused_kernel_metadata.cuda_task,
    #     (dest,),
    #     patch_info;
    #     blocks=(fused_kernel_metadata.blocks,),
    #     threads=(fused_kernel_metadata.threads,),
    #     prefix_scalars=(fused_kernel_metadata.ctx,),
    #     kernel_input_args_count=0,
    #     kernel_output_args_count=1,
    # )

    #! DO I NEED TO DO TYPE PROMOTION CHECKS??
    return dest
end
