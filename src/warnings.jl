### THE SCALAR INDEXING LOGIC IS COPIED FROM GPUArrays.jl ###

export allowdouble, @allowdouble, assertdouble, allowscalar, @allowscalar, assertscalar

@enum DoublePromotion PromotionAllowed PromotionWarn PromotionWarned PromotionDisallowed
@enum ScalarIndexing ScalarAllowed ScalarWarn ScalarWarned ScalarDisallowed

# if the user explicitly calls allowscalar, use that setting for all new tasks
# XXX: use context variables to inherit the parent task's setting, once available.
const requested_scalar_indexing = Ref{Union{Nothing,ScalarIndexing}}(nothing)
const requested_double_promotion = Ref{Union{Nothing,DoublePromotion}}(nothing)


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

@noinline function default_double_promotion()
    if isinteractive()
        # try to detect the REPL
        repl_task = repl_frontend_task()
        if repl_task isa Task
            if repl_task === current_task()
                # we always allow scalar iteration on the REPL's frontend task,
                # where we often trigger scalar indexing by displaying GPU objects.
                PromotionAllowed
            else
                PromotionDisallowed
            end
        else
            # we couldn't detect a REPL in this interactive session, so default to a warning
            PromotionWarn
        end
    else
        # non-interactively, we always disallow Promotion iteration
        PromotionDisallowed
    end
end

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
    assertdouble(op)

Assert that a certain operation `op` performs promotion to double. If this is not allowed, an
error will be thrown ([`allowdouble`](@ref)).
"""
function assertdouble(op, ::Type{FROM}, ::Type{TO}) where {FROM, TO}
    behavior = get(task_local_storage(), :DoublePromotion, nothing)
    if behavior === nothing
        behavior = requested_double_promotion[]
        if behavior === nothing
            behavior = default_double_promotion()
        end
        task_local_storage(:DoublePromotion, behavior)
    end

    behavior = behavior::DoublePromotion
    if behavior === PromotionAllowed
        # fast path
        return
    end

    _assertdouble(op, behavior, FROM, TO)
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

@noinline function _assertdouble(op, behavior, ::Type{FROM}, ::Type{TO}) where {FROM, TO}
    if behavior == PromotionDisallowed
        errordouble(op, FROM, TO)
    elseif behavior == PromotionWarn
        warndouble(op, FROM, TO)
        task_local_storage(:DoublePromotion, PromotionWarned)
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

function doubledesc(op, ::Type{FROM}, ::Type{TO}) where {FROM, TO}
    desc = """Invocation of $op resulted in implicit promotion of an NDArray from $(FROM) to 
              double precision: $(TO). This is typically caused by mixing NDArrays or literals 
              with different precision. Double precision typically slow on GPU and promotion
              to double precision was probably unintended and should be avoided.

              If you want to allow promotion to double precision, use `allowdouble` or `allowdouble`
              to enable promotion to double globally or for the operations in question. If all
              operations start in double precision no errors or warnings will trigger."""
end

@noinline function warnscalar(op)
    desc = scalardesc(op)
    @warn("""Performing scalar indexing on task $(current_task()).
             $desc""")
end

@noinline function warnsdouble(op, ::Type{FROM}, ::Type{TO}) where {FROM, TO}
    desc = doubledesc(op, FROM, TO)
    @warn("""Promotiong data to double precision on task $(current_task()).
             $desc""")
end

@noinline function errorscalar(op)
    desc = scalardesc(op)
    error("""Scalar indexing is disallowed.
             $desc""")
end

@noinline function errordouble(op, ::Type{FROM}, ::Type{TO}) where {FROM, TO}
    desc = doubledesc(op, FROM, TO)
    error("""Implicit promotion to double precision is disallowed.
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
    allowdouble([true])
    allowdouble([true]) do
        ...
    end

Use this function to allow or disallow promotion to double precision, either globally or for the
duration of the do block.

See also: [`@allowdouble`](@ref).
"""
allowdouble

function allowdouble(f::Base.Callable)
    task_local_storage(f, :DoublePromotion, PromotionAllowed)
end

function allowdouble(allow::Bool=true)
    if allow
        @warn """It's not recommended to use allowdouble([true]) to allow promotion to double precision.
                 Instead, use `allowdouble() do end` or `@allowdouble` to denote exactly which operations can convert to double precision.""" maxlog=1
    end
    setting = allow ? PromotionAllowed : PromotionDisallowed
    task_local_storage(:DoublePromotion, setting)
    requested_double_promotion[] = setting
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
    @allowdouble() begin
        # code that can use scalar indexing
    end

Denote which operations can use scalar indexing.

See also: [`allowdouble`](@ref).
"""
macro allowdouble(ex)
    quote
        local tls_value = get(task_local_storage(), :DoublePromotion, nothing)
        task_local_storage(:DoublePromotion, PromotionAllowed)
        @__tryfinally($(esc(ex)),
                      isnothing(tls_value) ? delete!(task_local_storage(), :DoublePromotion)
                                           : task_local_storage(:DoublePromotion, tls_value))
    end
end
