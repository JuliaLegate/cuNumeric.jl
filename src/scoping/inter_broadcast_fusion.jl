module InterBroadcastFusion

export rewrite_scope

using ..ScopingUtils

# Recombine single-use broadcast statements before lifetime analysis:
#
#   product = A .* B
#   C[:, :] = product .+ 2
#
# becomes:
#
#   C[:, :] .= A .* B .+ 2
#
# The pass is syntax-only and has no NDArray or cuNumeric dependencies.

# These records exist only during macro expansion, not during array execution.

# Array/scalar inputs read by an expression, plus names whose rebinding would
# change its meaning if evaluation is delayed until a later statement.
struct ReadDependencies
    inputs::Set{Symbol}
    bindings::Set{Symbol} # Also includes callable names that could be rebound.
end

# A statement such as `tmp = A .+ B` that produces a single-use temporary.
# Records its expression and the statement positions of its definition and use.
struct BroadcastProducer
    name::Symbol
    definition::Int
    consumer::Int
    expression::Any
end

# Temporaries to replace with their expressions at the use site, together with
# any runtime disjointness checks needed to make that delayed evaluation safe.
struct FusionPlan
    producers::Dict{Int,BroadcastProducer}
    alias_checks::Vector{Tuple{Symbol,Symbol}}
end
FusionPlan() = FusionPlan(Dict{Int,BroadcastProducer}(), Tuple{Symbol,Symbol}[])

# Substitutions accumulated while applying a plan, with original statement
# positions and before/after expressions used to report the rewrites.
struct RewriteState
    replacements::Dict{Symbol,Any}
    sources::Dict{Symbol,Vector{Int}}
    events::Vector{NamedTuple}
end
RewriteState() = RewriteState(Dict{Symbol,Any}(), Dict{Symbol,Vector{Int}}(), NamedTuple[])

# Only move arithmetic/indexing expressions across other statements. Unknown
# calls may mutate their arguments (even when their names do not end in `!`).
# This name-based allowlist assumes ordinary numerical methods; it does not
# prove that a shadowed function or an overloaded method is free of side effects.
const _READONLY_CALLS = Set((:+, :-, :*, :/, :^, :%, :fld, :cld, :mod, :rem,
    :(:), :abs, :abs2, :sqrt, :exp, :log, :sin, :cos, :tan, :inv, :min, :max,
    :ifelse, :iszero, :isfinite, :isnan, :identity, :size, :axes, :length,
    :firstindex, :lastindex, :eltype, :one, :zero, :<, :>, :<=, :>=, :(==), :(!=),
    :Bool, :Int, :Int8, :Int16, :Int32, :Int64, :UInt, :UInt8, :UInt16, :UInt32,
    :UInt64, :Float16, :Float32, :Float64, :ComplexF32, :ComplexF64))

_read_dependencies!(deps, ::Any) = false
_read_dependencies!(deps, ::Number) = true
_read_dependencies!(deps, ::QuoteNode) = true

function _read_dependencies!(deps, expr::Symbol)
    expr in (:end, :(:), :nothing, :true, :false) || push!(deps, expr)
    return true
end

function _read_dependencies!(deps, expr::Expr)
    reference = _reference(expr)
    if !isnothing(reference)
        return reference.array isa Symbol &&
            _read_dependencies!(deps, reference.array) &&
            all(x -> _read_dependencies!(deps, x), reference.indices)
    end
    # Parameter fields such as args.dt; writes to properties remain barriers.
    if expr.head === :. && length(expr.args) == 2 && expr.args[2] isa QuoteNode
        return _read_dependencies!(deps, expr.args[1])
    end
    call = _call(expr)
    isnothing(call) && (call = _dotcall(expr))
    isnothing(call) && return false
    f = call.f
    _is_broadcast_op(f) && (f = Symbol(chop(string(f); head=1, tail=0)))
    return f in _READONLY_CALLS && all(x -> _read_dependencies!(deps, x), call.args)
end

_is_readonly(expr) = _read_dependencies!(Set{Symbol}(), expr)

function _dependencies(expr)
    inputs = Set{Symbol}()
    _read_dependencies!(inputs, expr) || return nothing
    return ReadDependencies(inputs, Set(walk_symbols(expr)))
end

function _statement_assignment(stmt)
    assignment = _assignment(stmt)
    return isnothing(assignment) ? _broadcast_assignment(stmt) : assignment
end

# A simple destination whose indexing has no unknown effects. Property writes
# and calls that compute destinations are deliberately excluded.
function _write_target(lhs)
    lhs isa Symbol && return lhs
    reference = _reference(lhs)
    isnothing(reference) && return nothing
    reference.array isa Symbol || return nothing
    all(_is_readonly, reference.indices) || return nothing
    return reference.array
end

function _known_assignment(stmt)
    assignment = _statement_assignment(stmt)
    isnothing(assignment) && return false
    return !isnothing(_write_target(assignment.lhs)) && _is_readonly(assignment.rhs)
end

function _entry_guard_valid(stmts, write_index)
    # An earlier unknown call could replace storage after the entry check.
    return all(i -> _known_assignment(stmts[i]), 1:(write_index - 1))
end

function _write_checks(stmts, index, assignment, deps::ReadDependencies, guard_roots)
    _is_readonly(assignment.rhs) || return nothing
    target = _write_target(assignment.lhs)
    isnothing(target) && return nothing
    target in guard_roots || return nothing
    _entry_guard_valid(stmts, index) || return nothing
    # Entry checks can name stable arguments, not locals created later.
    issubset(deps.inputs, guard_roots) || return nothing
    target in deps.inputs && return nothing
    return [(target, input) for input in Base.sort!(collect(deps.inputs); by=string)]
end

function _delay_checks(stmts, producer::BroadcastProducer, rhs, guard_roots)
    deps = _dependencies(rhs)
    isnothing(deps) && return nothing
    checks = Tuple{Symbol,Symbol}[]
    for i in (producer.definition + 1):(producer.consumer - 1)
        assignment = _assignment(stmts[i])
        if !isnothing(assignment) && assignment.lhs isa Symbol
            assignment.lhs in deps.bindings && return nothing
            _is_readonly(assignment.rhs) || return nothing
            continue
        end
        isnothing(assignment) && (assignment = _broadcast_assignment(stmts[i]))
        isnothing(assignment) && return nothing
        write_checks = _write_checks(stmts, i, assignment, deps, guard_roots)
        isnothing(write_checks) && return nothing
        append!(checks, write_checks)
    end
    return checks
end

function _substitute_symbols(expr, replacements::Dict{Symbol,Any})
    assignment = _assignment(expr)
    isnothing(assignment) && return _replace_symbols(expr, replacements)
    assignment.lhs isa Symbol || return _replace_symbols(expr, replacements)
    rhs = _replace_symbols(assignment.rhs, replacements)
    return :($(assignment.lhs) = $rhs)
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
    Base.sort!(indices)
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

function _broadcast_consumer(stmt)
    assignment = _statement_assignment(stmt)
    rhs = isnothing(assignment) ? stmt : assignment.rhs
    _is_broadcast_syntax(rhs) && _is_readonly(rhs) || return false
    return isnothing(assignment) || !isnothing(_write_target(assignment.lhs))
end

function _producer(stmts, index, protected)
    assignment = _assignment(stmts[index])
    isnothing(assignment) && return nothing
    name, rhs = assignment.lhs, assignment.rhs
    name isa Symbol && _is_broadcast_syntax(rhs) || return nothing
    name in protected && return nothing
    consumer = _single_use_index(stmts, name, index)
    isnothing(consumer) && return nothing
    _broadcast_consumer(stmts[consumer]) || return nothing
    return BroadcastProducer(name, index, consumer, rhs)
end

function _plan_fusion(stmts, protected, guard_roots)
    plan = FusionPlan()
    expanded = Dict{Symbol,Any}()
    for index in eachindex(stmts)
        producer = _producer(stmts, index, protected)
        isnothing(producer) && continue
        # Include the original inputs of already-elided producers. Otherwise a
        # later rebind can become invisible through a chain like t -> s -> out.
        rhs = _replace_symbols(producer.expression, expanded)
        checks = _delay_checks(stmts, producer, rhs, guard_roots)
        isnothing(checks) && continue
        plan.producers[index] = producer
        expanded[producer.name] = rhs
        append!(plan.alias_checks, checks)
    end
    unique!(plan.alias_checks)
    return plan
end

function _record_producer!(state::RewriteState, producer::BroadcastProducer)
    indices = _source_indices(producer.expression, state.sources)
    push!(indices, producer.definition)
    state.sources[producer.name] = indices
    state.replacements[producer.name] =
        _substitute_symbols(producer.expression, state.replacements)
    return nothing
end

function _rewrite_consumer!(state::RewriteState, stmts, index)
    original = stmts[index]
    indices = _source_indices(original, state.sources)
    stmt = _substitute_symbols(original, state.replacements)
    isempty(indices) && return stmt

    stmt = _fuse_into_destination(stmt)
    before = Expr(:block, (stmts[i] for i in indices)..., original)
    push!(state.events, (; before, fused=stmt))
    return stmt
end

function _apply_plan(scope, stmts, plan::FusionPlan)
    state = RewriteState()
    rewritten = Any[]
    for index in eachindex(stmts)
        producer = get(plan.producers, index, nothing)
        if isnothing(producer)
            push!(rewritten, _rewrite_consumer!(state, stmts, index))
        else
            _record_producer!(state, producer)
        end
    end
    return Expr(scope.head, rewritten...), state.events
end

function _rewrite_scope(scope, protected, guard_roots)
    stmts = _scope_statements(scope)
    isnothing(stmts) && return scope, NamedTuple[], Tuple{Symbol,Symbol}[]
    plan = _plan_fusion(stmts, protected, guard_roots)
    rewritten, events = _apply_plan(scope, stmts, plan)
    return rewritten, events, plan.alias_checks
end

"""
    rewrite_scope(scope; on_rewrite=nothing, protected=Set{Symbol}()) -> scope

Fuse eligible single-use broadcast producers into their consumer and return the
rewritten scope. Symbols in `protected` — typically whatever the scope returns —
are never fused so they stay materialized. When provided, `on_rewrite` is called
with a named tuple containing the `before` and `fused` expressions for each
rewrite.

Nonadjacent producers may cross read-only bindings that do not rebind their
inputs. With `guard_roots` and an `alias_checks` collector, writes to stable
arguments may also be crossed: the caller must guard the returned rewrite with
disjointness checks for those `(destination, input)` pairs and provide a fallback.
"""
function rewrite_scope(scope; on_rewrite=nothing, protected=Set{Symbol}(),
    guard_roots=Set{Symbol}(), alias_checks=nothing)
    # Callers without a guard collector receive only statically safe rewrites.
    roots = isnothing(alias_checks) ? Set{Symbol}() : guard_roots
    rewritten, fusion_events, checks = _rewrite_scope(scope, protected, roots)
    isnothing(alias_checks) || append!(alias_checks, checks)
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
