module InterBroadcastFusion

export rewrite_scope

using ..ScopingUtils

function _substitute_symbols(expr, replacements::Dict{Symbol,Any})
    assignment = _assignment(expr)
    isnothing(assignment) && return _replace_symbols(expr, replacements)
    assignment.lhs isa Symbol || return _replace_symbols(expr, replacements)
    rhs = _replace_symbols(assignment.rhs, replacements)
    return :($(assignment.lhs) = $rhs)
end

function _indexed_assignment_base(stmt)
    assignment = _assignment(stmt)
    isnothing(assignment) && return nothing
    reference = _reference(assignment.lhs)
    isnothing(reference) && return nothing
    reference.array isa Symbol || return nothing
    return reference.array
end

function _safe_to_delay_broadcast(
    stmts, def_idx::Int, use_idx::Int, dependencies::Set{Symbol}, lazy_defs::Set{Int}
)
    for i in (def_idx + 1):(use_idx - 1)
        stmt = stmts[i]
        i in lazy_defs && continue

        # An indexed write to an unrelated array does not invalidate the lazy
        # producer. Any other intervening statement is conservatively a barrier.
        mutated = _indexed_assignment_base(stmt)
        if !isnothing(mutated) && !(mutated in dependencies)
            continue
        end
        return false
    end
    return true
end

function _single_use_index(stmts, symbol::Symbol, def_idx::Int)
    use_idx = nothing
    for i in (def_idx + 1):length(stmts)
        symbols = walk_symbols(stmts[i])
        occurrences = count(candidate -> candidate == symbol, symbols)
        occurrences == 0 && continue
        if occurrences != 1 || !isnothing(use_idx)
            return nothing
        end
        use_idx = i
    end
    return use_idx
end

function _source_indices(expr, replacement_sources)
    indices = Int[]
    for symbol in walk_symbols(expr)
        append!(indices, get(replacement_sources, symbol, Int[]))
    end
    unique!(indices)
    sort!(indices)
    return indices
end

function _fuse_into_destination(stmt)
    assignment = _assignment(stmt)
    isnothing(assignment) && return stmt
    _is_broadcast_syntax(assignment.rhs) || return stmt

    reference = _reference(assignment.lhs)
    isnothing(reference) && return stmt
    reference.array isa Symbol || return stmt

    if reference.array in walk_symbols(assignment.rhs)
        return stmt
    end
    return Expr(:(.=), assignment.lhs, assignment.rhs)
end

function _rewrite_scope(scope)
    stmts = _scope_statements(scope)
    isnothing(stmts) && return scope, NamedTuple[]

    definitions = Dict{Symbol,Tuple{Int,Any}}()
    lazy_defs = Set{Int}()
    for (i, stmt) in enumerate(stmts)
        assignment = _assignment(stmt)
        if !isnothing(assignment) && assignment.lhs isa Symbol &&
            _is_broadcast_syntax(assignment.rhs)
            definitions[assignment.lhs] = (i, assignment.rhs)
            push!(lazy_defs, i)
        end
    end

    inlineable = Dict{Symbol,Tuple{Int,Any}}()
    for (sym, (def_idx, rhs)) in definitions
        use_idx = _single_use_index(stmts, sym, def_idx)
        isnothing(use_idx) && continue
        dependencies = Set(walk_symbols(rhs))
        if !_safe_to_delay_broadcast(stmts, def_idx, use_idx, dependencies, lazy_defs)
            continue
        end
        inlineable[sym] = (def_idx, rhs)
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
            assignment = _assignment(original_stmt)
            source_indices = _source_indices(assignment.rhs, replacement_sources)
            push!(source_indices, i)
            replacement_sources[sym] = source_indices
            replacements[sym] = _substitute_symbols(inlineable[sym][2], replacements)
            continue
        end

        source_indices = _source_indices(original_stmt, replacement_sources)
        stmt = _substitute_symbols(original_stmt, replacements)

        if !isempty(source_indices)
            stmt = _fuse_into_destination(stmt)
            before = Expr(
                :block,
                (stmts[source_idx] for source_idx in source_indices)...,
                original_stmt,
            )
            push!(fusion_events, (; before, fused=stmt))
        end
        push!(rewritten, stmt)
    end

    return Expr(scope.head, rewritten...), fusion_events
end

"""
    rewrite_scope(scope; on_rewrite=nothing) -> scope

Fuse eligible single-use broadcast producers into their consumer and return the
rewritten scope. When provided, `on_rewrite` is called with a named tuple
containing the `before` and `fused` expressions for each rewrite.
"""
function rewrite_scope(scope; on_rewrite=nothing)
    rewritten, fusion_events = _rewrite_scope(scope)
    if !isnothing(on_rewrite)
        for event in fusion_events
            on_rewrite(event)
        end
    end
    return rewritten
end

function _print_expr(io::IO, expr)
    clean = _strip_lines(expr)
    rendered = sprint(Base.show_unquoted, clean)
    for line in eachline(IOBuffer(rendered))
        println(io, "    ", line)
    end
    return nothing
end

function log_rewrite(event; io::IO=stdout)
    println(io, "\n", "="^40, " inter-broadcast fusion rewrite")
    println(io, "  before")
    _print_expr(io, event.before)
    println(io, "  fused")
    _print_expr(io, event.fused)
    return nothing
end

end
