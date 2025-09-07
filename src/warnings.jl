### THE SCALAR INDEXING LOGIC IS COPIED FROM GPUArrays.jl ###

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
    _repl_frontend_task[]
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
        return
    end

    _assertscalar(op, behavior)
end

"""
    assertpromotion(op)

Assert that a certain operation `op` performs promotion to a wider type. If this is not allowed, an
error will be thrown ([`assertpromotion`](@ref)).
"""
function assertpromotion(op, ::Type{FROM}, ::Type{TO}) where {FROM, TO}
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
        return
    end

    _assertpromotion(op, behavior, FROM, TO)
end

@noinline function _assertscalar(op, behavior)
    if behavior == ScalarDisallowed
        errorscalar(op)
    elseif behavior == ScalarWarn
        warnscalar(op)
        task_local_storage(:ScalarIndexing, ScalarWarned)
    end

    return
end

@noinline function _assertpromotion(op, behavior, ::Type{FROM}, ::Type{TO}) where {FROM, TO}
    if behavior == PromotionDisallowed
        errordouble(op, FROM, TO)
    elseif behavior == PromotionWarn
        warndouble(op, FROM, TO)
        task_local_storage(:ImplicitPromotion, PromotionWarned)
    end

    return
end

function scalardesc(op)
    desc = """Invocation of $op resulted in scalar indexing of an NDArray.
              This is typically caused by calling an iterating implementation of a method.
              This is very slow and should be avoided.

              If you want to allow scalar iteration, use `allowscalar` or `@allowscalar`
              to enable scalar iteration globally or for the operations in question."""
end

function promotiondesc(op, ::Type{FROM}, ::Type{TO}) where {FROM, TO}
    desc = """Invocation of $op resulted in implicit promotion of an NDArray from $(FROM) to 
              wider type: $(TO). This is typically caused by mixing NDArrays or literals 
              with different precision. This can cause extra copies of data and is slow.

              If you want to allow implicit promotion to wider types, use `allowpromotion` or `@allowpromotion`
              to enable implicit promotion."""
end

@noinline function warnscalar(op)
    desc = scalardesc(op)
    @warn("""Performing scalar indexing on task $(current_task()).
             $desc""")
end

@noinline function warnsdouble(op, ::Type{FROM}, ::Type{TO}) where {FROM, TO}
    desc = promotiondesc(op, FROM, TO)
    @warn("""Promotiong data to wider type on task $(current_task()).
             $desc""")
end

@noinline function errorscalar(op)
    desc = scalardesc(op)
    error("""Scalar indexing is disallowed.
             $desc""")
end

@noinline function errordouble(op, ::Type{FROM}, ::Type{TO}) where {FROM, TO}
    desc = promotiondesc(op, FROM, TO)
    error("""Implicit promotion to wider type is disallowed.
             $desc""")
end

# Like a try-finally block, except without introducing the try scope
# NOTE: This is deprecated and should not be used from user logic. A proper solution to
# this problem will be introduced in https://github.com/JuliaLang/julia/pull/39217
macro __tryfinally(ex, fin)
    Expr(:tryfinally,
       :($(esc(ex))),
       :($(esc(fin)))
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
    task_local_storage(f, :ScalarIndexing, ScalarAllowed)
end

function allowscalar(allow::Bool=true)
    if allow
        @warn """It's not recommended to use allowscalar([true]) to allow scalar indexing.
                 Instead, use `allowscalar() do end` or `@allowscalar` to denote exactly which operations can use scalar operations.""" maxlog=1
    end
    setting = allow ? ScalarAllowed : ScalarDisallowed
    task_local_storage(:ScalarIndexing, setting)
    requested_scalar_indexing[] = setting
    return
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
    return
end

function allowpromotion(allow::Bool=true)
    if allow
        @warn """It's not recommended to use allowpromotion([true]) to allow promotion to double precision.
                 Instead, use `allowpromotion() do end` or `@allowpromotion` to denote exactly which operations can convert to double precision.""" maxlog=1
    end
    setting = allow ? PromotionAllowed : PromotionDisallowed
    task_local_storage(:ImplicitPromotion, setting)
    requested_implicit_promotion[] = setting
    return
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
                      isnothing(tls_value) ? delete!(task_local_storage(), :ScalarIndexing)
                                           : task_local_storage(:ScalarIndexing, tls_value))
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
                      isnothing(tls_value) ? delete!(task_local_storage(), :ImplicitPromotion)
                                           : task_local_storage(:ImplicitPromotion, tls_value))
    end
end
