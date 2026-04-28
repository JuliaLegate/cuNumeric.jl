
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

_isbits_size_str(::Type{T}) where {T} = isbitstype(T) ? string(sizeof(T)) : "n/a"

function _collect_cudevicearray_offsets!(offsets::Vector{Int}, ::Type{T}, base::Int=0) where {T}
    if T <: CUDACore.CuDeviceArray
        push!(offsets, base)
        return offsets
    end
    if isbitstype(T) && fieldcount(T) > 0
        for i in 1:fieldcount(T)
            FT = fieldtype(T, i)
            _collect_cudevicearray_offsets!(offsets, FT, base + Int(fieldoffset(T, i)))
        end
    end
    return offsets
end

function _collect_cudevicearray_offsets(::Type{T}) where {T}
    return Tuple(_collect_cudevicearray_offsets!(Int[], T, 0))
end

function _default_broadcast_threads(ndrange)
    n = prod(ndrange)
    return min(256, max(1, n))
end

function _cufunction_from_types(f, types_tuple_type; maxthreads=nothing, always_inline=false)
    return CUDACore.cufunction(
        f,
        types_tuple_type;
        kernel=true,
        maxthreads=maxthreads,
        always_inline=always_inline,
    )
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

    # if fancy thing doesnt work we can use this
    # threads = _default_broadcast_threads(ndrange)

    workgroupsize = CUDACore.CUDAKernels.threads_to_workgroupsize(threads, ndrange)
    iterspace, dynamic = KA.partition(obj, ndrange, workgroupsize)
    ctx = KA.mkcontext(obj, ndrange, iterspace)

    blocks = length(KA.blocks(iterspace))
    threads = length(KA.workitems(iterspace))
    blocks == 0 && return "", 0, 0, ctx

    buf = IOBuffer()
    CUDATools.code_ptx(buf, obj.f, (typeof(ctx), DEST_T, BC_T); raw=false, kernel=true)
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
    input_deps = Tuple(arg for arg in bc.args if arg isa NDArray)
    bc_gpu = CUDACore.cudaconvert(bc)
    bc_ndarray_offsets = _collect_cudevicearray_offsets(typeof(bc_gpu))
    @assert length(bc_ndarray_offsets) == length(input_deps)

    launch_broadcast(
        fused_kernel_metadata.cuda_task,
        input_deps,
        (dest,),
        (bc_gpu,);
        blocks=(fused_kernel_metadata.blocks,),
        threads=(fused_kernel_metadata.threads,),
        prefix_scalars=(fused_kernel_metadata.ctx,),
        kernel_input_args_count=0,
        kernel_output_args_count=1,
        bc_ndarray_offsets=bc_ndarray_offsets,
    )

    #! DOUBLE CHECK bc.args ACTAULLY GIVES BACK ARRAYS
    #! HOW TO GET SCALARS???
    # scalars = ????
    # launch(cuda_task, bc.args, (dest,), scalars; blocks, threads)

    #! DO I NEED TO DO TYPE PROMOTION CHECKS??

    return dest
end
