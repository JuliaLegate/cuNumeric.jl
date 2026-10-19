function sibling_group(p, position)
    id = p.schedule[position]
    op = p.operations[id]
    op.kind isa WriteOp{true} && op.expression.head === :(.=) || return nothing
    dest = first(op.expression.args)
    dest isa ValueRef || return nothing
    producer = p.values[dest.id].producer
    producer != 0 && p.operations[producer].kind isa IndexOp{true} || return nothing
    readonly_expression(p, last(op.expression.args)) || return nothing
    firstpos = position
    while firstpos > 1
        before = p.operations[p.schedule[firstpos - 1]]
        before.statement == op.statement || break
        allowed =
            before.kind isa Union{IndexOp,InputOp} ||
            (before.kind isa CallOp && isempty(before.inputs) && before.readonly)
        allowed && before.readonly || return nothing
        firstpos -= 1
    end
    hoists = p.schedule[firstpos:(position - 1)]
    producer in hoists || return nothing
    roots = unique([
        p.operations[i].kind.root for i in hoists if p.operations[i].kind isa IndexOp
    ])
    any(iszero, roots) && return nothing
    return (; id, firstpos, position, dest, hoists, roots)
end

replace_roots(kind::OperationKind, replacements) = kind
function replace_roots(kind::IndexOp{View}, replacements) where {View}
    return IndexOp{View}(get(replacements, kind.root, ValueRef(kind.root)).id)
end

function duplicate_hoists!(p, ids)
    replacements = Dict{Int,ValueRef}()
    operations = Dict{Int,Int}()
    for id in ids
        old = p.operations[id]
        outputs = Int[]
        newop = length(p.operations) + 1
        for v in old.outputs
            record = deepcopy(p.values[v])
            record.name = gensym(:read)
            record.producer = newop
            push!(p.values, record)
            add_vertex!(p.lifetimes)
            newvalue = length(p.values)
            for parent in inneighbors(p.lifetimes, v)
                add_edge!(p.lifetimes, get(replacements, parent, ValueRef(parent)).id, newvalue)
            end
            replacements[v] = ValueRef(newvalue)
            push!(outputs, newvalue)
        end
        term = replace_values(v -> get(replacements, v.id, v), old.expression)
        push!(
            p.operations,
            Operation(replace_roots(old.kind, replacements), term, operands(term), outputs,
                old.statement, old.readonly, old.destination, true),
        )
        operations[id] = newop
    end
    return operations, replacements
end

function sibling_run(p, after_statement)
    run = Any[]
    for position in eachindex(p.schedule)
        group = sibling_group(p, position)
        isnothing(group) && continue
        p.operations[group.id].statement > after_statement || continue
        if !isempty(run) && (last(run).position + 1 != group.firstpos ||
            any(g -> g.dest == group.dest, run))
            length(run) >= 2 && break
            empty!(run)
        end
        push!(run, group)
    end
    return run
end

function sibling_fast(p, run, prefix)
    q = clone(p)
    dests = Expr(:tuple, [g.dest for g in run]...)
    calls = Expr(:tuple, [broadcasted(last(p.operations[g.id].expression.args)) for g in run]...)
    term = :(cuNumeric.copyto_fused_siblings!($dests, $calls))
    fused = fused_operation!(q, term, [g.id for g in run])
    q.schedule = [prefix; fused; p.schedule[(last(run).position + 1):end]]
    return rebuild!(q)
end

function replace_operands!(p, ids, replacements)
    replace(v) = get(replacements, v.id, v)
    for id in ids
        op = p.operations[id]
        op.expression = replace_values(replace, op.expression)
        op.kind = replace_roots(op.kind, replacements)
        for output in op.outputs, parent in collect(inneighbors(p.lifetimes, output))
            haskey(replacements, parent) || continue
            rem_edge!(p.lifetimes, parent, output)
            add_edge!(p.lifetimes, replacements[parent].id, output)
        end
    end
    p.result = replace_values(replace, p.result)
    return p
end

# Recreate speculative copied reads at their original execution positions.
# Unused speculative handles are released by this branch's lifetime pass.
function sibling_slow(p, run, prefix)
    q = clone(p)
    laterhoists = reduce(vcat, [g.hoists for g in run[2:end]])
    opmap, replacements = duplicate_hoists!(q, laterhoists)
    remainder = p.schedule[first(run).position:end]
    replace_operands!(q, setdiff(remainder, laterhoists), replacements)
    q.schedule = [prefix; [get(opmap, id, id) for id in remainder]]
    return rebuild!(q)
end

function freshness_guard(p, run)
    checks = Any[]
    for k in 2:length(run), id in run[k].hoists
        read = p.operations[id]
        read.kind isa IndexOp{false} || continue
        for previous in run[1:(k - 1)]
            push!(
                checks,
                :(cuNumeric._hoisted_read_fresh($(previous.dest),
                    $(ValueRef(read.kind.root)), $(ValueRef(only(read.outputs))))),
            )
        end
    end
    return isempty(checks) ? true : reduce((a, b) -> Expr(:&&, a, b), checks)
end

function sibling_guard(p, run, before)
    roots = unique(reduce(vcat, [g.roots for g in run]))
    terms = [
        p.values[v].producer in before ? ValueRef(v) :
        something(p.values[v].external, ValueRef(v)) for v in roots
    ]
    return :(cuNumeric._all_ndarrays($(terms...)))
end

function fuse_siblings(p; after_statement=0)
    p.schedule = filter(i -> p.operations[i].active, p.schedule)
    run = sibling_run(p, after_statement)
    length(run) >= 2 || return p
    before = p.schedule[1:(first(run).firstpos - 1)]
    prefix = [before; reduce(vcat, [g.hoists for g in run])]
    laststatement = p.operations[last(run).id].statement
    continue_fusion(q) = fuse_siblings(q; after_statement=laststatement)
    inner = Choice(prefix, freshness_guard(p, run),
        continue_fusion(sibling_fast(p, run, prefix)),
        continue_fusion(sibling_slow(p, run, prefix)))
    return Choice(before, sibling_guard(p, run, before), inner, continue_fusion(clone(p)))
end
