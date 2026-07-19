
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
) where {DEST_T}
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

            return FusedBroadcastMetadata(ctx, threads, blocks, cuda_task)
        end
    end
end

# Fused-broadcast introspection. Set `cuNumeric.BCAST_FUSION_DEBUG[] = true` to
# dump each kernel's expr/inputs/scalars/launch geometry before launch.
const BCAST_FUSION_DEBUG = Ref(false)

# Handle keyed on the same objectid the kernel uses to dedup inputs. Slice-assign
# slices come from `nda_get_slice` with no `parent`, so object identity is the
# only overlap the printer can see (the "aliases output" flag is same-object only).
_nd_tag(arr::NDArray) = string("NDArray#", string(objectid(arr) % 0x1000000; base=16, pad=6))
_nd_describe(arr::NDArray{T}) where {T} = string(_nd_tag(arr), " ", join(shape(arr), "x"), " ::", T)

_fname(f::Function) = string(nameof(f))
_fname(@nospecialize(f)) = string(f)

# Reconstruct the op tree; call before `flatten`, while nesting mirrors the source.
_bcast_tree_str(x::NDArray) = _nd_tag(x)
_bcast_tree_str(x::Base.Broadcast.Extruded) = _nd_tag(x.x)
_bcast_tree_str(x::Number) = repr(x)
_bcast_tree_str(x::Base.RefValue) = string("^", repr(x[]))
_bcast_tree_str(x) = string("<", typeof(x), ">")
function _bcast_tree_str(bc::Base.Broadcast.Broadcasted)
    return string(_fname(bc.f), ".(", join(_bcast_tree_str.(bc.args), ", "), ")")
end

# Recover the kernel's plain name
function _demangle_head(s::AbstractString)
    m = match(r"^_Z([0-9]+)(.*)$", s)
    m === nothing && return s
    return first(m.captures[2], parse(Int, m.captures[1]))
end

# `arg_map` records the kernel's argument order (0=output, ≥1=input idx+1,
# <0=scalar idx); reconstruct it as a readable `output -> name(args...)` call.
function _kernel_signature(
    dest::NDArray,
    unique_ndarrays::AbstractVector{<:NDArray},
    actual_scalars::AbstractVector,
    arg_map::AbstractVector{<:Integer},
    func::AbstractString,
)
    token(a::Integer) =
        if a == 0
            "output"
        elseif a > 0
            _nd_tag(unique_ndarrays[a])
        else
            repr(actual_scalars[-a])
        end
    call_args = [token(a) for a in arg_map if a != 0]
    return string(_nd_tag(dest), " -> ", _demangle_head(func), "(", join(call_args, ", "), ")")
end

function _describe_fused_broadcast(
    dest, tree_str, unique_ndarrays, actual_scalars, static_args, arg_map, fkm, ndrange
)
    io = IOBuffer()
    field(k, v) = println(io, "  ", rpad(k, 8), v)
    println(io, "\n", "="^40, " fused broadcast kernel")
    field("expr", tree_str)
    field("output", _nd_describe(dest))
    field("inputs", "$(length(unique_ndarrays)) unique NDArray(s)")
    for (i, nd) in enumerate(unique_ndarrays)
        alias = objectid(nd) == objectid(dest) ? "  (aliases output)" : ""
        println(io, "    [", i - 1, "] ", _nd_describe(nd), alias)
    end
    isempty(actual_scalars) || field("scalars", join(repr.(actual_scalars), ", "))
    isempty(static_args) || field("static", join(repr.(static_args), ", "))
    field("arg_map", "$(Int.(arg_map))  (0=output, >=1=input idx+1, <0=scalar)")
    field("launch", "$(fkm.blocks) blocks x $(fkm.threads) threads, ndrange=$ndrange")
    field(
        "call",
        _kernel_signature(dest, unique_ndarrays, actual_scalars, arg_map, fkm.cuda_task.func),
    )
    print(String(take!(io)))
    return nothing
end

# Pre-launch promotion policy for fused broadcast. Mirrors unfused
# `unravel_broadcast_tree` (`__checked_promote_op` / `__my_promote_type` per
# tree node) and the dest-side widen check of `checked_promote_arr` in
# `_copyto_unfused!`. Fused writes `dest` in place, so there is no post-fuse
# promote / `nda_move` — this must run before kernel launch.
#
# The tree walk is typed on `Broadcasted{S,Ax,F,Args}` and the concrete
# `Args` Tuple, so Julia specializes/inlines per broadcast shape. Same-eltype
# trees (e.g. all Float32) constant-fold `is_wider_type` to false and DCE the
# `assertpromotion` calls; residual cost is only the specialize/inline frame.
@inline function _assert_fused_broadcast_promotion(
    dest::NDArray{DT}, bc::B
) where {DT,B<:Base.Broadcast.Broadcasted}
    T_OUT = _assert_fused_broadcast_tree(bc)
    is_wider_type(DT, T_OUT) && assertpromotion(promote_type, T_OUT, DT)
    return nothing
end

# Leaf NDArray / Number / RefValue / etc.
@inline _assert_fused_broadcast_tree(x) = eltype(x)

# Nested node: recurse through typed Args, then op/input promote checks.
@inline function _assert_fused_broadcast_tree(
    bc::Base.Broadcast.Broadcasted{S,Ax,F,Args}
) where {S,Ax,F,Args}
    eltypes = _fused_checked_eltypes(bc.args)
    T_OUT = __checked_promote_op(bc.f, eltypes)
    __my_promote_type(eltypes.parameters...)
    return T_OUT
end

# Typed Tuple walk — method selection replaces runtime `isa` / Any iteration.
# Returns `Type{Tuple{...}}` for `__checked_promote_op(f, ::Type{Tuple{...}})`.
@inline _fused_checked_eltypes(::Tuple{}) = Tuple{}

@inline function _fused_checked_eltypes(args::Tuple{A}) where {A}
    T1 = _assert_fused_broadcast_tree(getfield(args, 1))
    return Tuple{T1}
end

@inline function _fused_checked_eltypes(args::Tuple{A,B}) where {A,B}
    T1 = _assert_fused_broadcast_tree(getfield(args, 1))
    T2 = _assert_fused_broadcast_tree(getfield(args, 2))
    return Tuple{T1,T2}
end

# literal_pow and other ternary broadcast args
@inline function _fused_checked_eltypes(args::Tuple{A,B,C}) where {A,B,C}
    T1 = _assert_fused_broadcast_tree(getfield(args, 1))
    T2 = _assert_fused_broadcast_tree(getfield(args, 2))
    T3 = _assert_fused_broadcast_tree(getfield(args, 3))
    return Tuple{T1,T2,T3}
end

@inline function _fused_checked_eltypes(args::Tuple)
    T1 = _assert_fused_broadcast_tree(getfield(args, 1))
    rest = _fused_checked_eltypes(Base.tail(args))
    return Tuple{T1,rest.parameters...}
end

function fuse_broadcast_tree!(dest::D, bc::B) where {D<:NDArray,B<:Base.Broadcast.Broadcasted}
    # Promotion checks use the pre-flatten tree (same shape as unfused unravel).
    _assert_fused_broadcast_promotion(dest, bc)

    # Capture the readable tree before flatten collapses the nesting.
    tree_str = BCAST_FUSION_DEBUG[] ? _bcast_tree_str(bc) : ""

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

    BCAST_FUSION_DEBUG[] && _describe_fused_broadcast(
        dest, tree_str, unique_ndarrays, actual_scalars, static_args, arg_map, fkm, ndrange
    )

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

    # Fused kernel already wrote `dest` in place; promotion was checked pre-launch.
    return dest
end
