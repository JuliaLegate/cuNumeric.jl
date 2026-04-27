
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

const _BCAST_PTX_CACHE = Dict{Tuple{Any,DataType,DataType,Any},Tuple{CUDATask,Int,Int}}()
const _BCAST_PTX_CACHE_LOCK = ReentrantLock()

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
    blocks == 0 && return "", 0, 0

    buf = IOBuffer()
    CUDATools.code_ptx(buf, obj.f, (typeof(ctx), DEST_T, BC_T); raw=false, kernel=true)
    return String(take!(buf)), threads, blocks
end

function get_cuda_task(
    obj::KA.Kernel{CUDACore.CUDAKernels.CUDABackend},
    ::Type{DEST_T},
    ::Type{BC_T};
    ndrange,
) where {DEST_T,BC_T}
    key = (obj, DEST_T, BC_T, ndrange)
    lock(_BCAST_PTX_CACHE_LOCK) do
        # Also stores in cache if not found in Dict
        return get!(_BCAST_PTX_CACHE, key) do
            ptx, blocks, threads = get_ptx(obj, DEST_T, BC_T; ndrange=ndrange)
            func_name = extract_kernel_name(_ptx)
            ptx_task(ptx, func_name)
            (CUDATask(func_name, (DEST_T, BC_T)), blocks, threads)
        end
    end
end

function fuse_broadcast_tree!(dest::D, bc::B) where {D<:NDArray,B<:Base.Broadcast.Broadcasted}
    bc = Base.Broadcast.preprocess(dest, bc)

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

    # Create "fake" type signatures to compile kernel with CUDA.jl infrastructure
    DEST_T = map_cuda_type(typeof(dest))
    BC_T = map_cuda_type(B)

    # Lookup in cache, if not found, compile and cache
    cuda_task, threads, blocks = get_cuda_task(broadcast_kernel, DEST_T, BC_T; ndrange=ndrange)
    # TODO: register with Legate runtime like @cuda_task does

    # https://github.com/JuliaGPU/CUDA.jl/blob/345c1600ebd561135148bb04ee2657f521a40e25/CUDACore/src/CUDAKernels.jl#L111

    #! DO I NEED TO DO TYPE PROMOTION CHECKS??

    return dest
end
