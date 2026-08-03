
struct RuntimeBroadcastArg{J} end
struct StaticBroadcastArg{J} end

Base.@propagate_inbounds @inline function _gpu_broadcast_getindex(x, I)
    return @inbounds Base.Broadcast._broadcast_getindex(x, I)
end

Base.@propagate_inbounds _gpu_broadcast_getindex(x::Number, I) = x

# Same-shaped fused operands already use the destination index.
Base.@propagate_inbounds @inline function _gpu_broadcast_getindex(
    x::CuStridedDeviceArray, I::Union{Integer,CartesianIndex}
)
    return @inbounds x[I]
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

@inline function _broadcast_linear_work_id()
    return (Int(CUDACore.blockIdx().x) - 1) * Int(CUDACore.blockDim().x) +
           Int(CUDACore.threadIdx().x)
end

function make_linear_kernel(dest, bc::Base.Broadcast.Broadcasted, arg_plan, static_args)
    f = bc.f

    @kernel unsafe_indices = true function broadcast_kernel_linear_splat(dest, runtime_args...)
        I = _broadcast_linear_work_id()
        if I <= length(dest)
            @inbounds args_modified = _materialize_broadcast_args(
                arg_plan, runtime_args, static_args, I
            )
            @inbounds dest[I] = Base.Broadcast._broadcast_getindex_evalf(f, args_modified...)
        end
    end

    return broadcast_kernel_linear_splat
end

@inline function _broadcast_cartesian_work_id()
    row =
        (Int(CUDACore.blockIdx().y) - 1) * Int(CUDACore.blockDim().y) +
        Int(CUDACore.threadIdx().y)
    col =
        (Int(CUDACore.blockIdx().x) - 1) * Int(CUDACore.blockDim().x) +
        Int(CUDACore.threadIdx().x)
    return CartesianIndex(row, col)
end

function make_cartesian_kernel(dest, bc::Base.Broadcast.Broadcasted, arg_plan, static_args)
    f = bc.f

    @kernel unsafe_indices = true function broadcast_kernel_cartesian_splat(
        dest, runtime_args...
    )
        I = _broadcast_cartesian_work_id()
        if I[1] <= size(dest, 1) && I[2] <= size(dest, 2)
            @inbounds args_modified = _materialize_broadcast_args(
                arg_plan, runtime_args, static_args, I
            )
            @inbounds dest[I] = Base.Broadcast._broadcast_getindex_evalf(f, args_modified...)
        end
    end

    return broadcast_kernel_cartesian_splat
end

@inline function _broadcast_cartesian_work_id_3d()
    i =
        (Int(CUDACore.blockIdx().z) - 1) * Int(CUDACore.blockDim().z) +
        Int(CUDACore.threadIdx().z)
    j =
        (Int(CUDACore.blockIdx().y) - 1) * Int(CUDACore.blockDim().y) +
        Int(CUDACore.threadIdx().y)
    k =
        (Int(CUDACore.blockIdx().x) - 1) * Int(CUDACore.blockDim().x) +
        Int(CUDACore.threadIdx().x)
    return CartesianIndex(i, j, k)
end

function make_cartesian_kernel_3d(
    dest, bc::Base.Broadcast.Broadcasted, arg_plan, static_args
)
    f = bc.f

    @kernel unsafe_indices = true function broadcast_kernel_cartesian_3d_splat(
        dest, runtime_args...
    )
        I = _broadcast_cartesian_work_id_3d()
        if I[1] <= size(dest, 1) && I[2] <= size(dest, 2) && I[3] <= size(dest, 3)
            @inbounds args_modified = _materialize_broadcast_args(
                arg_plan, runtime_args, static_args, I
            )
            @inbounds dest[I] = Base.Broadcast._broadcast_getindex_evalf(f, args_modified...)
        end
    end

    return broadcast_kernel_cartesian_3d_splat
end

function make_broadcast_kernel(
    dest::NDArray{<:Any,2}, bc::Base.Broadcast.Broadcasted, arg_plan, static_args
)
    return make_cartesian_kernel(dest, bc, arg_plan, static_args)
end

function make_broadcast_kernel(
    dest::NDArray{<:Any,3}, bc::Base.Broadcast.Broadcasted, arg_plan, static_args
)
    return make_cartesian_kernel_3d(dest, bc, arg_plan, static_args)
end

function make_broadcast_kernel(dest, bc::Base.Broadcast.Broadcasted, arg_plan, static_args)
    return make_linear_kernel(dest, bc, arg_plan, static_args)
end

struct FusedBroadcastMetadata
    ctx::Any # KA.CompilerMetadata (compilation / arg layout; not global ndrange)
    threads::Int # occupancy thread budget; device chooses final launch dims
    cuda_task::CUDATask
end

# Cache by kernel identity + arg types. Launch geometry is derived per-GPU from
# the local PhysicalArray, so global ndrange is not a key.
const _BCAST_PTX_CACHE = Dict{Tuple{Any,DataType,DataType},FusedBroadcastMetadata}()
const _BCAST_PTX_CACHE_LOCK = ReentrantLock()

# Every NDArray leaf must match `dest` shape. Shape-mismatched broadcasts
# (e.g. matrix .+ vector) are handled by the unfused path.
#
# Slice views are allowed: RunPTXBroadcastTask packs their element strides.
@inline _can_fuse_linear_broadcast_leaf(dest, ::Number) = true
@inline _can_fuse_linear_broadcast_leaf(dest, ::Base.RefValue) = true
@inline function _can_fuse_linear_broadcast_leaf(dest, x::NDArray)
    return size(x) == size(dest)
end
@inline function _can_fuse_linear_broadcast_leaf(dest, x::Base.Broadcast.Extruded)
    return _can_fuse_linear_broadcast_leaf(dest, x.x)
end
@inline function _can_fuse_linear_broadcast_leaf(dest, bc::Base.Broadcast.Broadcasted)
    return _can_fuse_linear_broadcast_args(dest, bc.args)
end
@inline _can_fuse_linear_broadcast_leaf(dest, @nospecialize(x)) = false

@inline _can_fuse_linear_broadcast_args(dest, ::Tuple{}) = true
@inline function _can_fuse_linear_broadcast_args(dest, args::Tuple)
    return _can_fuse_linear_broadcast_leaf(dest, getfield(args, 1)) &&
           _can_fuse_linear_broadcast_args(dest, Base.tail(args))
end

"""
Return true when same-shaped broadcast fusion is safe for `bc` into `dest`.

Requires every NDArray leaf to have the same size as `dest`. Scalars / RefValues
are allowed. Unknown leaf types refuse fusion (fall back to unfused).

Also refuses 0-d destinations: `RunPTXBroadcastTask` only supports dims in
`[1, 6]`; 0-d falls back to the unfused path.
"""
@inline function can_fuse_linear_broadcast(dest::NDArray, bc::Base.Broadcast.Broadcasted)
    # Device-side launch dims require at least one dimension.
    ndims(dest) >= 1 || return false
    return _can_fuse_linear_broadcast_leaf(dest, bc)
end

# Same-shaped operands do not need Broadcast's dynamic index projection.
@inline function _unwrap_fusion_arg(x::Base.Broadcast.Extruded)
    if all(x.keeps)
        return x.x
    end
    throw(
        ArgumentError(
            "Broadcast fusion does not support shape-mismatched " *
            "Extruded arguments; use the unfused broadcast path",
        ),
    )
end
@inline _unwrap_fusion_arg(x) = x
@inline _unwrap_fusion_args(args::Tuple) = _unwrap_fusion_arg.(args)

# Host-side scalar alignment for fusion — same as unfused
# `T_IN = __my_promote_type(...); unchecked_promote_arr.(args, T_IN)`, but
# `unchecked_promote_scalar` keeps Numbers as scalars for the PTX arg buffer.
#
# Must run on the host *before* `get_cuda_task` so the PTX cache key sees the
# promoted scalar types. `a .^ 2` is unaffected: `Val{N}` is static; op-aware
# checks stay in pre-flatten `_assert_fused_broadcast_promotion`.
function _align_fused_runtime_args(runtime_args::Tuple)
    isempty(runtime_args) && return runtime_args
    T_IN = __my_promote_type(map(eltype, runtime_args)...)
    return map(a -> unchecked_promote_scalar(a, T_IN), runtime_args)
end

cudevice_array_offset(::Type{T}) where {T<:CuStridedDeviceArray} = 0
cudevice_array_offset(::Type{T}) where {T<:Base.Broadcast.Extruded} = Int(fieldoffset(T, 1))

stores_cudevicearray(::Type{T}) where {T<:CuStridedDeviceArray} = true
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

"""
Host-side occupancy probe: return a thread *budget* and a minimal KA `ctx` for
PTX compilation. Final blocks/threads are chosen in `RunPTXBroadcastTask` from
each GPU's local `PhysicalArray` shape.
"""
function _threads_from_occupancy(
    obj::KA.Kernel{CUDACore.CUDAKernels.CUDABackend},
    ::Type{DEST_T},
    ARG_TYPES...;
    ndrange=(1024,),
) where {DEST_T}
    backend = KA.backend(obj)

    # Compile-time iterspace only — not used for multi-GPU launch coverage.
    ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(obj, ndrange, nothing)
    ctx = KA.mkcontext(obj, ndrange, iterspace)

    maxthreads =
        if KA.workgroupsize(obj) <: KA.StaticSize
            prod(KA.get(KA.workgroupsize(obj)))
        else
            nothing
        end

    tt = Base.to_tuple_type((typeof(ctx), DEST_T, ARG_TYPES...))
    host_kernel = CUDACore.cufunction(
        obj.f,
        tt;
        kernel=true,
        maxthreads=maxthreads,
        always_inline=backend.always_inline,
    )
    config = CUDACore.launch_configuration(host_kernel.fun; max_threads=prod(ndrange))
    threads = Int(config.threads)

    # Bake ctx workitems to the occupancy budget so KA metadata stays consistent
    # with the thread count we pass to the device as a budget.
    workgroupsize = CUDACore.CUDAKernels.threads_to_workgroupsize(threads, ndrange)
    iterspace, dynamic = KA.partition(obj, ndrange, workgroupsize)
    ctx = KA.mkcontext(obj, ndrange, iterspace)
    threads = length(KA.workitems(iterspace))

    return threads, ctx
end

"""
    get_ptx(obj, DEST_T, arg_types...) -> (ptx, threads, ctx)

Compile a KA CUDA kernel using types only and choose an occupancy thread budget.
"""
function get_ptx(
    obj::KA.Kernel{CUDACore.CUDAKernels.CUDABackend},
    ::Type{DEST_T},
    arg_types...;
) where {DEST_T}
    threads, ctx = _threads_from_occupancy(obj, DEST_T, arg_types...)
    threads == 0 && return "", 0, ctx

    buf = IOBuffer()
    # dump_module=true keeps only_entry=false so linked libdevice helpers (e.g. cos
    # slowpaths) are not emptied into unresolved .externs before cuModuleLoad.
    CUDATools.code_ptx(buf, obj.f, (typeof(ctx), DEST_T, arg_types...);
        raw=false, dump_module=true, kernel=true, ptx=v"7.8")

    return String(take!(buf)), threads, ctx
end

function get_cuda_task(
    obj::KA.Kernel{CUDACore.CUDAKernels.CUDABackend},
    dest::D,
    runtime_args::RT,
) where {D<:NDArray,RT<:Tuple}
    DEST_T = map_cuda_type(D)
    ARG_TYPES = map_cuda_type.(typeof.(runtime_args))

    key = (obj, D, RT)

    lock(_BCAST_PTX_CACHE_LOCK) do
        return get!(_BCAST_PTX_CACHE, key) do
            ptx, threads, ctx = get_ptx(obj, DEST_T, ARG_TYPES...)

            orig_name = extract_kernel_name(ptx)
            unique_name = orig_name * "_" * string(hash(ptx); base=16)
            ptx = replace(ptx, orig_name => unique_name)

            ptx_task(ptx, unique_name)
            cuda_task = CUDATask(unique_name, (DEST_T, ARG_TYPES...))

            return FusedBroadcastMetadata(ctx, threads, cuda_task)
        end
    end
end

# Fused-broadcast introspection. Set `cuNumeric.BCAST_FUSION_DEBUG[] = true` to
# dump each kernel's expr/inputs/scalars/launch geometry before launch.
const BCAST_FUSION_DEBUG = Ref(false)

_fname(f::Function) = string(nameof(f))
_fname(@nospecialize(f)) = string(f)

# Reconstruct the op tree; call before `flatten`, while nesting mirrors the source.
function _bcast_tree_str(leaf_name, bc::Base.Broadcast.Broadcasted)
    args = (_bcast_tree_str(leaf_name, arg) for arg in bc.args)
    return string(_fname(bc.f), "(", join(args, ", "), ")")
end
_bcast_tree_str(leaf_name, x::Base.Broadcast.Extruded) = _bcast_tree_str(leaf_name, x.x)
_bcast_tree_str(leaf_name, x) = leaf_name(x)

function _bcast_tree_str(bc::Base.Broadcast.Broadcasted)
    return _bcast_tree_str(bc) do x
        x isa NDArray && return "NDArray"
        x isa Number && return repr(x)
        x isa Base.RefValue && return string("^", repr(x[]))
        return string("<", typeof(x), ">")
    end
end

function _bcast_runtime_tree_str(
    bc::Base.Broadcast.Broadcasted, ndarray_to_input_idx, actual_scalars=nothing
)
    scalar_idx = 0
    return _bcast_tree_str(bc) do x
        if x isa NDArray
            input_idx = get(ndarray_to_input_idx, objectid(x), nothing)
            return input_idx === nothing ? "NDArray" : "input{$input_idx}"
        end

        if x isa Number
            idx = scalar_idx
            scalar_idx += 1
            value = isnothing(actual_scalars) ? x : actual_scalars[idx + 1]
            return repr(value)
        end

        if x isa Base.RefValue
            value = x[]
            if value isa Number
                idx = scalar_idx
                scalar_idx += 1
                runtime_value =
                    isnothing(actual_scalars) ? value : actual_scalars[idx + 1]
                return repr(runtime_value)
            end
            return repr(value)
        end

        return string("<", typeof(x), ">")
    end
end

function _bcast_scope_name(
    bc::Base.Broadcast.Broadcasted, ndarray_to_input_idx, actual_scalars=nothing
)
    return string(
        "broadcast.",
        _bcast_runtime_tree_str(bc, ndarray_to_input_idx, actual_scalars),
    )
end

# Recover the kernel's plain name
function _demangle_head(s::AbstractString)
    m = match(r"^_Z([0-9]+)(.*)$", s)
    m === nothing && return s
    return first(m.captures[2], parse(Int, m.captures[1]))
end

# `arg_map` records the kernel's argument order (0=output, ≥1=input idx+1,
# <0=scalar idx). Scalar slots are replaced with their values in the signature.
function _kernel_arg_name(a::Integer)
    return if a == 0
        "output"
    elseif a > 0
        "input{$(a - 1)}"
    else
        "scalar{$(-a - 1)}"
    end
end

function _kernel_signature(
    dest::NDArray,
    unique_ndarrays::AbstractVector{<:NDArray},
    actual_scalars::AbstractVector,
    arg_map::AbstractVector{<:Integer},
    func::AbstractString,
)
    token(a) = a < 0 ? repr(actual_scalars[-a]) : _kernel_arg_name(a)
    call_args = [token(a) for a in arg_map if a != 0]
    return string("broadcast.", _demangle_head(func), "(", join(call_args, ", "), ")")
end

function _ndarray_debug_summary(nd::NDArray)
    summary = "NDArray{$(eltype(nd)), $(ndims(nd))} $(size(nd))"
    if _is_ndarray_slice(nd)
        return "$summary slice, parent $(size(nd.parent))"
    end
    return summary
end

function _describe_fused_broadcast(
    dest, tree_str, unique_ndarrays, actual_scalars, static_args, arg_map, fkm, ndrange
)
    io = IOBuffer()
    field(k, v) = println(io, "  ", rpad(k, 8), v)
    println(io, "\n", "="^40, " fused broadcast kernel")
    field("expr", tree_str)
    field("output", _ndarray_debug_summary(dest))
    field("inputs", "input{N} ($(length(unique_ndarrays)) unique)")
    println(io, "    ", rpad("N", 4), "value")
    for (i, nd) in enumerate(unique_ndarrays)
        alias = objectid(nd) == objectid(dest) ? "  (aliases output)" : ""
        println(io, "    ", rpad(string(i - 1), 4), _ndarray_debug_summary(nd), alias)
    end
    isempty(static_args) || field("static", join(repr.(static_args), ", "))
    indexing = ndims(dest) in (2, 3) ? "cartesian" : "linear"
    field(
        "launch",
        "host thread budget=$(fkm.threads), indexing=$indexing, " *
        "blocks=device(local tile), global_ndrange=$ndrange",
    )
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
    bc_scope = bc

    bc = Base.Broadcast.preprocess(dest, bc)
    bc = Base.Broadcast.instantiate(bc)
    bc = Base.Broadcast.flatten(bc)

    # Things like exponentiation generate arguments like Base.RefValue
    # which do not work with our pattern for making CUDA kernels as they are
    # not is-bits types. We split these out manually into static args and handle
    # them separately from runtime args (i.e., arrays, scalars)
    runtime_args, static_args, arg_plan = split_broadcast_args_for_kernel(
        _unwrap_fusion_args(bc.args)
    )
    # Host-only, before PTX cache: same `__my_promote_type` + Number convert as
    # unfused `T_IN` / `unchecked_promote_arr`. Kernel sees already-aligned types.
    runtime_args = _align_fused_runtime_args(runtime_args)

    broadcast_kernel = make_broadcast_kernel(dest, bc, arg_plan, static_args)

    ndrange = ndims(dest) > 0 ? size(dest) : (1,)

    bck_cuda = broadcast_kernel(CUDACore.CUDAKernels.CUDABackend())

    fkm = get_cuda_task(bck_cuda, dest, runtime_args)

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

    if BCAST_FUSION_DEBUG[]
        tree_str = _bcast_runtime_tree_str(
            bc_scope, ndarray_to_input_idx, actual_scalars
        )
        _describe_fused_broadcast(
            dest,
            tree_str,
            unique_ndarrays,
            actual_scalars,
            static_args,
            arg_map,
            fkm,
            ndrange,
        )
    end

    @task_scope _bcast_scope_name(bc_scope, ndarray_to_input_idx, actual_scalars) begin
        # `blocks=1` is a placeholder; RunPTXBroadcastTask overwrites grid dims
        # from the local PhysicalArray. `threads` is only the occupancy budget (tx).
        # Scalars after ctx: num_kernel_args, arg_map...
        launch(
            fkm.cuda_task,
            input_ndarrays,
            (dest,),
            (Int32(length(arg_map)), arg_map..., actual_scalars...);
            blocks=1,
            threads=fkm.threads,
            taskid=cuNumeric.RUN_PTX_BROADCAST,
            ctx=fkm.ctx,
            validate_shapes=false,
        )
    end

    # Fused kernel already wrote `dest` in place; promotion was checked pre-launch.
    return dest
end
