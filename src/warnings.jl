export allowpromotion, @allowpromotion, assertpromotion, allowscalar, @allowscalar, assertscalar

@enum ImplicitPromotion PromotionAllowed PromotionWarn PromotionWarned PromotionDisallowed
@enum ScalarIndexing ScalarAllowed ScalarWarn ScalarWarned ScalarDisallowed

# if the user explicitly calls allowscalar, use that setting for all new tasks
# XXX: use context variables to inherit the parent task's setting, once available.
const requested_scalar_indexing = Ref{Union{Nothing,ScalarIndexing}}(nothing)
const requested_implicit_promotion = Ref{Union{Nothing,ImplicitPromotion}}(nothing)

const _repl_frontend_task = Ref{Union{Nothing,Missing,Task}}()
function repl_frontend_task()
    if !isassigned(_repl_frontend_task)
        _repl_frontend_task[] = get_repl_frontend_task()
    end
    return _repl_frontend_task[]
end
@noinline function get_repl_frontend_task()
    if isdefined(Base, :active_repl)
        Base.active_repl.frontend_task
    else
        missing
    end
end

@noinline function default_scalar_indexing()
    if isinteractive()
        # try to detect the REPL
        repl_task = repl_frontend_task()
        if repl_task isa Task
            if repl_task === current_task()
                # we always allow scalar iteration on the REPL's frontend task,
                # where we often trigger scalar indexing by displaying GPU objects.
                ScalarAllowed
            else
                ScalarDisallowed
            end
        else
            # we couldn't detect a REPL in this interactive session, so default to a warning
            ScalarWarn
        end
    else
        # non-interactively, we always disallow scalar iteration
        ScalarDisallowed
    end
end

default_implicit_promotion() = PromotionDisallowed

"""
    assertscalar(op::String)

Assert that a certain operation `op` performs scalar indexing. If this is not allowed, an
error will be thrown ([`allowscalar`](@ref)).
"""
function assertscalar(op::String)
    behavior = get(task_local_storage(), :ScalarIndexing, nothing)
    if behavior === nothing
        behavior = requested_scalar_indexing[]
        if behavior === nothing
            behavior = default_scalar_indexing()
        end
        task_local_storage(:ScalarIndexing, behavior)
    end

    behavior = behavior::ScalarIndexing
    if behavior === ScalarAllowed
        # fast path
        return nothing
    end

    return _assertscalar(op, behavior)
end

"""
    assertpromotion(op)

Assert that a certain operation `op` performs promotion to a wider type. If this is not allowed, an
error will be thrown ([`assertpromotion`](@ref)).
"""
function assertpromotion(op, ::Type{FROM}, ::Type{TO}) where {FROM,TO}
    behavior = get(task_local_storage(), :ImplicitPromotion, nothing)
    if behavior === nothing
        behavior = requested_implicit_promotion[]
        if behavior === nothing
            behavior = default_implicit_promotion()
        end
        task_local_storage(:ImplicitPromotion, behavior)
    end

    behavior = behavior::ImplicitPromotion
    if behavior === PromotionAllowed
        # fast path
        return nothing
    end

    return _assertpromotion(op, behavior, FROM, TO)
end

@noinline function _assertscalar(op, behavior)
    if behavior == ScalarDisallowed
        errorscalar(op)
    elseif behavior == ScalarWarn
        warnscalar(op)
        task_local_storage(:ScalarIndexing, ScalarWarned)
    end

    return nothing
end

@noinline function _assertpromotion(op, behavior, ::Type{FROM}, ::Type{TO}) where {FROM,TO}
    if behavior == PromotionDisallowed
        errordouble(op, FROM, TO)
    elseif behavior == PromotionWarn
        warndouble(op, FROM, TO)
        task_local_storage(:ImplicitPromotion, PromotionWarned)
    end

    return nothing
end

const _CUNUMERIC_MODULE = @__MODULE__

# Sentinel for stack frames with no recoverable module. Prefer this over `nothing`
# so `_module_of_stackframe` is type-stable as `Module`. Must not match Base /
# LinearAlgebra / cuNumeric checks below.
const _UNKNOWN_STACK_MODULE = Module(:__cuNumeric_unknown_stack_module__, false, false)

@inline function _is_cunumeric_module(m::Module)
    m === _UNKNOWN_STACK_MODULE && return false
    m === _CUNUMERIC_MODULE && return true
    pm = parentmodule(m)
    while pm !== m
        pm === _CUNUMERIC_MODULE && return true
        m = pm
        pm = parentmodule(m)
    end
    return false
end

@inline function _is_linalg_module(m::Module)
    m === _UNKNOWN_STACK_MODULE && return false
    m === LinearAlgebra && return true
    nameof(m) === :LinearAlgebra && return true
    pm = parentmodule(m)
    while pm !== m
        (pm === LinearAlgebra || nameof(pm) === :LinearAlgebra) && return true
        m = pm
        pm = parentmodule(m)
    end
    return false
end

# Note: LinearAlgebra (and other stdlibs) often have parentmodule === Base, so callers
# must check `_is_linalg_module` before treating a frame as Base.
@inline function _is_base_module(m::Module)
    m === _UNKNOWN_STACK_MODULE && return false
    m === Base && return true
    nameof(m) === :Base && return true
    pm = parentmodule(m)
    while pm !== m
        (pm === Base || nameof(pm) === :Base) && return true
        m = pm
        pm = parentmodule(m)
    end
    return false
end

_module_from_def(def::Method) = def.module
_module_from_def(def::Module) = def
_module_from_def(_) = _UNKNOWN_STACK_MODULE

_module_of_linfo(linfo::Core.MethodInstance) = _module_from_def(linfo.def)
_module_of_linfo(linfo::Method) = linfo.module
# Julia 1.12+ often stores a CodeInstance on stack frames; unwrap to MethodInstance.
_module_of_linfo(linfo::Core.CodeInstance) = _module_of_linfo(linfo.def)
_module_of_linfo(_) = _UNKNOWN_STACK_MODULE

# When `frame.linfo` is missing (common for inlined frames), recover module from
# the source path so LinearAlgebra callers are not skipped and the walk does not
# fall through to loader frames like `Base.include_string`.
function _module_from_file(file)
    (file === nothing || file === :none) && return _UNKNOWN_STACK_MODULE
    f = string(file)
    # Match stdlib path segments; avoid false positives on user paths when possible.
    if occursin(r"(?:^|[/\\])LinearAlgebra(?:[/\\]|$)", f)
        return LinearAlgebra
    end
    if occursin(r"(?:^|[/\\])cuNumeric(?:\.jl)?(?:[/\\]|$)", f)
        return _CUNUMERIC_MODULE
    end
    # Base frames commonly appear as `./abstractarray.jl`, `./set.jl`, etc.
    if startswith(f, "./") || occursin(r"(?:^|[/\\])[Bb]ase(?:[/\\]|$)", f)
        return Base
    end
    return _UNKNOWN_STACK_MODULE
end

function _module_of_stackframe(frame::Base.StackTraces.StackFrame)
    m = _module_of_linfo(frame.linfo)
    m !== _UNKNOWN_STACK_MODULE && return m
    return _module_from_file(frame.file)
end

# Keyword bodies often look like `#cholesky!#272`; surface `cholesky!`.
function _clean_stack_func_name(fname)
    fname_sym = ifelse(fname isa Symbol, fname, Symbol(string(fname)))
    s = string(fname_sym)
    m = match(r"^#([^#]+)#\d+$", s)
    return m === nothing ? fname_sym : Symbol(m.captures[1])
end

# Frames that are never the user-facing "triggering" API for enrichment:
# - loaders / client entry (`include_string` / Julia 1.12 `IncludeInto` from
#   `include`ing tests, etc.)
# - keyword-call wrappers (`kwcall`) that would otherwise outrank cholesky/svd
# - AbstractArray iteration/indexing plumbing between NDArray getindex and the
#   real stdlib caller (e.g. LinearAlgebra.cholesky / Base.unique)
const _SKIP_STACK_FUNCS = Set{Symbol}((
    :include_string,
    :include,
    :include_relative,
    :_include,
    :IncludeInto, # Julia 1.12+ callable include wrapper (Base.IncludeInto)
    :eval,
    :exec_options,
    :_start,
    :invokelatest,
    :error,
    :stacktrace,
    :kwcall, # Base keyword-call wrapper; do not steal blame from cholesky/svd
    :iterate,
    :getindex,
    :setindex!,
    :indexed_iterate,
    Symbol("macro expansion"),
    Symbol("top-level scope"),
))

"""
Best-effort: outermost Base or LinearAlgebra frame above cuNumeric scalar-index
frames. Walk innermost-first, skip cuNumeric/Core (and frames with unknown
module), indexing/iteration plumbing, keyword-call wrappers (`kwcall`), and
Base loader frames (`include_string`, `IncludeInto` on Julia 1.12+, etc.).
Keep updating the candidate while still in Base/LinearAlgebra (last one wins)
so attribution names the user-facing API (`LinearAlgebra.svd`) rather than an
inner helper (`Base.lt`) or a loader (`Base.include_string`). Stop at the first
user/other-package frame and return that candidate, or `nothing` for the plain
message (e.g. user `Main`). Check LinearAlgebra before Base — stdlibs often
parent to Base.
"""
@noinline function _scalar_indexing_stdlib_caller()
    caller = nothing
    for frame in stacktrace()
        m = _module_of_stackframe(frame)
        # Skip frames with unknown module (same as previous `nothing` skip).
        m === _UNKNOWN_STACK_MODULE && continue
        (m === Core || _is_cunumeric_module(m)) && continue
        clean_name = _clean_stack_func_name(frame.func)
        clean_name in _SKIP_STACK_FUNCS && continue
        if _is_linalg_module(m)
            caller = (:LinearAlgebra, clean_name)
        elseif _is_base_module(m)
            caller = (:Base, clean_name)
        else
            # User / other package code — keep the last stdlib candidate, if any.
            break
        end
    end
    return caller
end

# Returns (enriched::Bool, desc::String). Enriched = Base or LinearAlgebra stdlib caller.
function scalardesc(op)
    caller = _scalar_indexing_stdlib_caller()
    if caller !== nothing
        modname, fname = caller
        # Base/LinearAlgebra AbstractArray fallback — name the outer API first.
        # No "Scalar indexing is disallowed." header (not part of the enriched template).
        return true,
        "`$modname.$fname` fell back to an AbstractArray implementation, which scalar-indexed an `NDArray`. " *
        "This $modname path is probably not implemented yet for `NDArray`. " *
        "Using `allowscalar` or `@allowscalar` might allow this function to work slowly, but it has not been tested."
    end

    # Plain user-level scalar indexing (unchanged).
    return false, """Invocation of $op resulted in scalar indexing of an `NDArray`.
              This is typically caused by calling an iterating implementation of a method.
              This is very slow and should be avoided. This can also happen if an external
              method (i.e., LinearAlgebra.kron) is not re-implemented in cuNumeric.jl. Because
              `NDArray`s subtype `AbstractArray`, the method call will dispatch to the
              `AbstractArray` implementation, which often iterates over the array.

              If you want to allow scalar iteration, use `allowscalar` or `@allowscalar`
              to enable scalar iteration globally or for the operations in question."""
end

function promotiondesc(op, ::Type{FROM}, ::Type{TO}) where {FROM,TO}
    return desc = """Invocation of $op resulted in implicit promotion of an NDArray from $(FROM) to
                     wider type: $(TO). This is typically caused by mixing NDArrays or literals
                     with different precision. This can cause extra copies of data and is slow.

                     If you want to allow implicit promotion to wider types, use `allowpromotion` or `@allowpromotion`
                     to enable implicit promotion."""
end

@noinline function warnscalar(op)
    _, desc = scalardesc(op)
    @warn("""Performing scalar indexing on task $(current_task()).
             $desc""")
end

@noinline function warnsdouble(op, ::Type{FROM}, ::Type{TO}) where {FROM,TO}
    desc = promotiondesc(op, FROM, TO)
    @warn("""Promotiong data to wider type on task $(current_task()).
             $desc""")
end

@noinline function errorscalar(op)
    enriched, desc = scalardesc(op)
    if enriched
        error(desc)
    else
        # Plain path keeps the historical disallow header.
        error("""Scalar indexing is disallowed.
                 $desc""")
    end
end

@noinline function errordouble(op, ::Type{FROM}, ::Type{TO}) where {FROM,TO}
    desc = promotiondesc(op, FROM, TO)
    error("""Implicit promotion to wider type is disallowed.
             $desc""")
end

# Like a try-finally block, except without introducing the try scope
# NOTE: This is deprecated and should not be used from user logic. A proper solution to
# this problem will be introduced in https://github.com/JuliaLang/julia/pull/39217
macro __tryfinally(ex, fin)
    return Expr(:tryfinally,
        :($(esc(ex))),
        :($(esc(fin))),
    )
end

"""
    allowscalar([true])
    allowscalar([true]) do
        ...
    end

Use this function to allow or disallow scalar indexing, either globall or for the
duration of the do block.

See also: [`@allowscalar`](@ref).
"""
allowscalar

function allowscalar(f::Base.Callable)
    return task_local_storage(f, :ScalarIndexing, ScalarAllowed)
end

function allowscalar(allow::Bool=true)
    if allow
        @warn """It's not recommended to use allowscalar([true]) to allow scalar indexing.
                 Instead, use `allowscalar() do end` or `@allowscalar` to denote exactly which operations can use scalar operations.""" maxlog=1
    end
    setting = allow ? ScalarAllowed : ScalarDisallowed
    task_local_storage(:ScalarIndexing, setting)
    requested_scalar_indexing[] = setting
    return nothing
end

"""
    allowpromotion([true])
    allowpromotion([true]) do
        ...
    end

Use this function to allow or disallow promotion to double precision, either globally or for the
duration of the do block.

See also: [`@allowpromotion`](@ref).
"""
allowpromotion

function allowpromotion(f::Base.Callable, allow::Bool=true)
    setting = allow ? PromotionAllowed : PromotionDisallowed
    task_local_storage(f, :ImplicitPromotion, setting)
    return nothing
end

function allowpromotion(allow::Bool=true)
    if allow
        @warn """It's not recommended to use allowpromotion([true]) to allow promotion to double precision.
                 Instead, use `allowpromotion() do end` or `@allowpromotion` to denote exactly which operations can convert to double precision.""" maxlog=1
    end
    setting = allow ? PromotionAllowed : PromotionDisallowed
    task_local_storage(:ImplicitPromotion, setting)
    requested_implicit_promotion[] = setting
    return nothing
end

"""
    @allowscalar() begin
        # code that can use scalar indexing
    end

Denote which operations can use scalar indexing.

See also: [`allowscalar`](@ref).
"""
macro allowscalar(ex)
    quote
        local tls_value = get(task_local_storage(), :ScalarIndexing, nothing)
        task_local_storage(:ScalarIndexing, ScalarAllowed)
        @__tryfinally($(esc(ex)),
            if isnothing(tls_value)
                delete!(task_local_storage(), :ScalarIndexing)
            else
                task_local_storage(:ScalarIndexing, tls_value)
            end)
    end
end

"""
    @allowpromotion() begin
        # code that can use scalar indexing
    end

Denote which operations can use scalar indexing.

See also: [`allowpromotion`](@ref).
"""
macro allowpromotion(ex)
    quote
        local tls_value = get(task_local_storage(), :ImplicitPromotion, nothing)
        task_local_storage(:ImplicitPromotion, PromotionAllowed)
        @__tryfinally($(esc(ex)),
            if isnothing(tls_value)
                delete!(task_local_storage(), :ImplicitPromotion)
            else
                task_local_storage(:ImplicitPromotion, tls_value)
            end)
    end
end
