function last_uses(p)
    lastuse = zeros(Int, length(p.values))
    for (position, id) in enumerate(p.schedule)
        op = p.operations[id]
        op.active || continue
        for v in Iterators.flatten((op.outputs, op.inputs))
            lastuse[v] = max(lastuse[v], position)
        end
    end
    for (id, value) in enumerate(p.values)
        value.deferred && (lastuse[id] = length(p.schedule))
    end
    return lastuse
end

function extend_lifetimes!(p, lastuse, escapes)
    for value in reverse(topological_sort_by_dfs(p.lifetimes))
        for parent in inneighbors(p.lifetimes, value)
            lastuse[parent] = max(lastuse[parent], lastuse[value])
            escapes[parent] |= escapes[value]
        end
    end
    return nothing
end

function cleanup(p)
    lastuse = last_uses(p)
    escapes = [v.escapes for v in p.values]
    extend_lifetimes!(p, lastuse, escapes)
    deletes = [Int[] for _ in p.schedule]
    for (id, value) in enumerate(p.values)
        value.owned && value.materialized && !escapes[id] && lastuse[id] > 0 || continue
        push!(deletes[lastuse[id]], id)
    end
    return deletes
end

expression(p, term) = replace_values(v -> p.values[v.id].name, term)
expression(p, term::OpaqueSyntax) = term.expression

operation_expression(p, op::Operation{BindOp}) = expression(p, op.expression)
function operation_expression(p, op::Operation)
    return Expr(:(=), p.values[only(op.outputs)].name, expression(p, op.expression))
end
function operation_expression(p, op::Operation{FusedOp})
    names = [p.values[v].name for v in op.outputs]
    return Expr(:(=), Expr(:tuple, names...), expression(p, op.expression))
end

struct ScopeBoundary
    header::Any
end

# Scope boundaries are graph operations, not inferred from emitted syntax.
# Folding from the end preserves Julia's sequential let-header visibility.
function emitted_block(items)
    body = Any[]
    for item in reverse(items)
        if item isa ScopeBoundary
            body = Any[Expr(:let, item.header, Expr(:block, reverse(body)...))]
        else
            push!(body, item)
        end
    end
    return Expr(:block, reverse(body)...)
end

emit_operation(p, op::Operation{ScopeOp}) = ScopeBoundary(expression(p, op.expression))
function emit_operation(p, op::Operation)
    ex = operation_expression(p, op)
    line = get(p.locations, op.statement, nothing)
    return isnothing(line) ? ex : Expr(:block, line, ex)
end

delete_expr(p, id) = :(cuNumeric.maybe_insert_delete($(p.values[id].name)))

function emit(p::Program; skip=0, finalizers=true)
    deletes = finalizers ? cleanup(p) : [Int[] for _ in p.schedule]
    out = Any[]
    # A shared prefix is deliberately emitted without cleanup. Values unused
    # on this branch can now be released; the other branch has its own plan.
    for position in 1:skip, id in deletes[position]
        push!(out, delete_expr(p, id))
    end
    for position in (skip + 1):length(p.schedule)
        op = p.operations[p.schedule[position]]
        op.active && push!(out, emit_operation(p, op))
        append!(out, [delete_expr(p, id) for id in deletes[position]])
    end
    push!(out, expression(p, p.result))
    return emitted_block(out)
end

function emit(choice::Choice; skip=0, finalizers=true)
    p = program(choice.fast)
    prefix = Any[
        emit_operation(p, p.operations[id]) for id in choice.prefix[(skip + 1):end]
        if p.operations[id].active
    ]
    count = length(choice.prefix)
    fast = emit(choice.fast; skip=count, finalizers)
    slow = emit(choice.slow; skip=count, finalizers)
    push!(prefix, Expr(:if, expression(p, choice.condition), fast, slow))
    return emitted_block(prefix)
end

function compile(scope; caller=parentmodule(@__MODULE__), protected=Set{Symbol}(),
    locals=Set{Symbol}(), policy=LocalScope(), fusion=true,
    aggressive=false, on_rewrite=nothing)
    p = parse_scope(scope; caller, protected, locals, policy)
    return emit(optimize(p; fusion, aggressive, on_rewrite))
end

# Syntax-level inspection entry point, also used by the fusion regression tests.
function rewrite_scope(scope; protected=Set{Symbol}(), guard_roots=Set{Symbol}(),
    alias_checks=nothing, on_rewrite=nothing,
    caller=parentmodule(@__MODULE__))
    roots = union(protected, guard_roots)
    locals = union(roots, parentmodule(@__MODULE__)._assigned_symbols(scope))
    p = parse_scope(scope; protected=roots, locals, caller, policy=VisibleScope())
    checks = fuse_producers!(p; guarded=(!isnothing(alias_checks)), on_rewrite)
    isnothing(alias_checks) || append!(alias_checks,
        [(p.values[a].external, p.values[b].external) for (a, b) in checks])
    return emit(p; finalizers=false)
end

function log_rewrite(event; io::IO=stdout)
    println(io, "\n", "="^40, " inter-broadcast fusion rewrite")
    for (label, ex) in (("before", event.before), ("fused", event.fused))
        println(io, "  ", label)
        for line in split(sprint(Base.show_unquoted, _strip_lines(ex)), '\n')
            println(io, "    ", line)
        end
    end
end
