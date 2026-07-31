module InterBroadcastFusion

export rewrite_scope

import ..cuNumeric: walk_symbols

_is_broadcast_op(op) = op isa Symbol && startswith(string(op), ".")

function _is_broadcast_syntax(expr)
    return expr isa Expr &&
           (
        (expr.head == :call && !isempty(expr.args) && _is_broadcast_op(expr.args[1])) ||
        (expr.head == :. && length(expr.args) == 2)
    )
end

function _symbol_occurrences(expr, target::Symbol)
    expr === target && return 1
    expr isa Expr || return 0
    return sum(arg -> _symbol_occurrences(arg, target), expr.args; init=0)
end

function _substitute_symbols(expr, replacements::Dict{Symbol,Any})
    expr isa Symbol && return get(replacements, expr, expr)
    expr isa Expr || return expr

    # A symbol assignment introduces/redefines its LHS; only substitute in RHS.
    if expr.head == :(=) && expr.args[1] isa Symbol
        return Expr(
            :(=), expr.args[1], _substitute_symbols(expr.args[2], replacements)
        )
    end
    return Expr(
        expr.head, (_substitute_symbols(arg, replacements) for arg in expr.args)...
    )
end

function _indexed_assignment_base(stmt)
    if stmt isa Expr && stmt.head == :(=) &&
        stmt.args[1] isa Expr && stmt.args[1].head == :ref
        base = stmt.args[1].args[1]
        return base isa Symbol ? base : nothing
    end
    return nothing
end

function _safe_to_delay_broadcast(
    stmts, def_idx::Int, use_idx::Int, dependencies::Set{Symbol}, lazy_defs::Set{Int}
)
    for i in (def_idx + 1):(use_idx - 1)
        stmt = stmts[i]
        stmt isa LineNumberNode && continue
        i in lazy_defs && continue

        # An indexed write to an unrelated array does not invalidate the lazy
        # producer. Any other intervening statement is conservatively a barrier.
        mutated = _indexed_assignment_base(stmt)
        mutated !== nothing && !(mutated in dependencies) && continue
        return false
    end
    return true
end

function _rewrite_scope(scope)
    Meta.isexpr(scope, (:block, :begin)) || return scope, NamedTuple[]
    stmts = collect(scope.args)

    definitions = Dict{Symbol,Tuple{Int,Any}}()
    lazy_defs = Set{Int}()
    for (i, stmt) in enumerate(stmts)
        if stmt isa Expr && stmt.head == :(=) && stmt.args[1] isa Symbol &&
            _is_broadcast_syntax(stmt.args[2])
            definitions[stmt.args[1]] = (i, stmt.args[2])
            push!(lazy_defs, i)
        end
    end

    inlineable = Dict{Symbol,Tuple{Int,Int,Any}}()
    for (sym, (def_idx, rhs)) in definitions
        use_indices = Int[]
        occurrences = 0
        for i in (def_idx + 1):length(stmts)
            count = _symbol_occurrences(stmts[i], sym)
            if count > 0
                occurrences += count
                push!(use_indices, i)
            end
        end
        occurrences == 1 || continue
        use_idx = only(use_indices)
        dependencies = Set(walk_symbols(rhs))
        _safe_to_delay_broadcast(stmts, def_idx, use_idx, dependencies, lazy_defs) ||
            continue
        inlineable[sym] = (def_idx, use_idx, rhs)
    end

    replacements = Dict{Symbol,Any}()
    replacement_sources = Dict{Symbol,Vector{Int}}()
    removed = Set(first(info) for info in values(inlineable))
    def_symbols = Dict(info[1] => sym for (sym, info) in inlineable)
    fusion_events = NamedTuple[]
    rewritten = Any[]

    for (i, original_stmt) in enumerate(stmts)
        if i in removed
            sym = def_symbols[i]
            source_indices = Int[]
            for dependency in walk_symbols(original_stmt.args[2])
                haskey(replacement_sources, dependency) || continue
                append!(source_indices, replacement_sources[dependency])
            end
            unique!(source_indices)
            sort!(source_indices)
            push!(source_indices, i)
            replacement_sources[sym] = source_indices
            replacements[sym] = _substitute_symbols(inlineable[sym][3], replacements)
            continue
        end

        source_indices = Int[]
        for dependency in walk_symbols(original_stmt)
            haskey(replacement_sources, dependency) || continue
            append!(source_indices, replacement_sources[dependency])
        end
        unique!(source_indices)
        sort!(source_indices)

        stmt = _substitute_symbols(original_stmt, replacements)

        # A materialized indexed assignment with a broadcast RHS normally
        # allocates a temporary and copies it into the slice. If the destination
        # base is absent from the RHS, write the fused tree directly to the view.
        direct_write = false
        if stmt isa Expr && stmt.head == :(=) &&
            stmt.args[1] isa Expr && stmt.args[1].head == :ref &&
            _is_broadcast_syntax(stmt.args[2])
            base = stmt.args[1].args[1]
            if base isa Symbol && !(base in Set(walk_symbols(stmt.args[2])))
                stmt = Expr(:(.=), stmt.args[1], stmt.args[2])
                direct_write = true
            end
        end

        if !isempty(source_indices) || direct_write
            before = Expr(
                :block,
                (deepcopy(stmts[source_idx]) for source_idx in source_indices)...,
                deepcopy(original_stmt),
            )
            push!(fusion_events, (; before, fused=deepcopy(stmt)))
        end
        push!(rewritten, stmt)
    end

    return Expr(scope.head, rewritten...), fusion_events
end

const _LOG_FUNCTION = GlobalRef(@__MODULE__, :maybe_log_rewrite)
const _DEBUG_FLAG = GlobalRef(parentmodule(@__MODULE__), :BCAST_FUSION_DEBUG)

function _log_call(event)
    before = QuoteNode(event.before)
    fused = QuoteNode(event.fused)
    enabled = Expr(:ref, _DEBUG_FLAG)
    return Expr(:call, _LOG_FUNCTION, enabled, before, fused)
end

"""
    rewrite_scope(scope) -> scope

Fuse eligible single-use broadcast producers into their consumer and return the
rewritten scope. Runtime log hooks are included for rewrites and are controlled
by `cuNumeric.BCAST_FUSION_DEBUG[]`.
"""
function rewrite_scope(scope)
    rewritten, fusion_events = _rewrite_scope(scope)
    isempty(fusion_events) && return rewritten
    log_calls = (_log_call(event) for event in fusion_events)
    return Expr(rewritten.head, log_calls..., rewritten.args...)
end

function is_log_call(expr)
    return Meta.isexpr(expr, :call) && expr.args[1] == _LOG_FUNCTION
end

function _print_expr(io::IO, expr)
    clean = expr isa Expr ? Base.remove_linenums!(deepcopy(expr)) : expr
    rendered = sprint(Base.show_unquoted, clean)
    for line in eachline(IOBuffer(rendered))
        println(io, "    ", line)
    end
    return nothing
end

function maybe_log_rewrite(enabled::Bool, before, fused; io::IO=stdout)
    enabled || return nothing
    println(io, "\n", "="^40, " inter-broadcast fusion rewrite")
    println(io, "  before")
    _print_expr(io, before)
    println(io, "  fused")
    _print_expr(io, fused)
    return nothing
end

end
