# A guarded rewrite owns two complete graph plans. Shared prefixes are emitted
# once, with each branch's lifetime pass accounting for their values separately.
struct Choice
    prefix::Vector{Int}
    condition::Any
    fast::Any
    slow::Any
end
program(p::Program) = p
program(p::Choice) = program(p.fast)

function uses(p, value)
    return [
        (id, count(==(value), p.operations[id].inputs)) for id in p.schedule
        if p.operations[id].active && !(p.operations[id].kind isa BindOp) &&
            value in p.operations[id].inputs
    ]
end

storage_roots(p, value) = storage_roots(p, p.operations[p.values[value].producer], value)
storage_roots(p, ::Operation, value) = [value]
function storage_roots(p, op::Operation{<:IndexOp}, value)
    return op.kind.root == 0 ? [value] : storage_roots(p, op.kind.root)
end
function storage_roots(p, op::Operation{<:BorrowingOp}, value)
    return unique(reduce(vcat, (storage_roots(p, i) for i in op.inputs); init=Int[]))
end

function delay_checks(p, producer, consumer; guarded=false)
    schedule = p.schedule
    lo, hi = findfirst(==(producer), schedule), findfirst(==(consumer), schedule)
    reads = unique(
        reduce(vcat, (storage_roots(p, i) for i in p.operations[producer].inputs); init=Int[])
    )
    checks = Tuple{Int,Int}[]
    for id in schedule[(lo + 1):(hi - 1)]
        op = p.operations[id]
        if op.kind isa BindOp
            # Retain the current fusion boundary at source-level rebinding,
            # even though value IDs also keep the previous value addressable.
            name = first(op.expression.args)
            any(v -> p.values[v].external === name, reads) && return nothing
        end
        (!op.active || op.readonly) && continue
        op.kind isa WriteOp || return nothing
        guarded || return nothing
        root = op.destination
        root != 0 && p.values[root].external in p.protected || return nothing
        all(v -> p.values[v].external in p.protected, reads) || return nothing
        root in reads && return nothing
        # Entry guards cannot be moved above unknown calls that might replace
        # array storage, even if that call's result is unused.
        all(j -> p.operations[j].readonly || p.operations[j].kind isa WriteOp,
            schedule[1:(findfirst(==(id), schedule) - 1)]) || return nothing
        append!(checks, ((root, v) for v in reads))
    end
    return checks
end

function fuse_destination!(p, consumer)
    op = p.operations[consumer]
    op.kind isa BroadcastOp || return nothing
    value = only(op.outputs)
    p.values[value].escapes && return nothing
    consumers = uses(p, value)
    length(consumers) == 1 || return nothing
    target = first(only(consumers))
    write = p.operations[target]
    write.kind isa WriteOp{false} || return nothing
    isnothing(delay_checks(p, consumer, target)) && return nothing
    lhs = first(write.expression.args)
    lhs isa Expr && lhs.head === :ref && readonly_expression(p, lhs) || return nothing
    write.destination in reduce(vcat, (storage_roots(p, v) for v in op.inputs); init=Int[]) &&
        return nothing
    # Only a broadcast which absorbed a producer is rewritten into its indexed
    # destination, matching the existing optimization's allocation boundary.
    slice = length(p.operations) + 1
    dest = length(p.values) + 1
    push!(p.values, Value(gensym(:destination), slice, true, false, false, true, nothing))
    add_vertex!(p.lifetimes)
    push!(
        p.operations,
        Operation(IndexOp{true}(value_id(first(lhs.args))), :(Base.@view $lhs), operands(lhs),
            [dest], write.statement, true, 0, true),
    )
    term = Expr(:(.=), ValueRef(dest), op.expression)
    p.operations[target] = Operation(WriteOp{true}(), term, operands(term), write.outputs,
        write.statement, false, write.destination, true)
    for output in write.outputs
        for parent in collect(inneighbors(p.lifetimes, output))
            rem_edge!(p.lifetimes, parent, output)
        end
        add_edge!(p.lifetimes, dest, output)
    end
    insert!(p.schedule, findfirst(==(target), p.schedule), slice)
    return eliminate!(p, op)
end

function eliminate!(p, op)
    op.active = false
    for value in op.outputs
        p.values[value].materialized = false
        for binding in p.operations
            binding.kind isa BindOp && value in binding.inputs && (binding.active = false)
        end
    end
    return nothing
end

function fuse_producers!(p; guarded=false, on_rewrite=nothing)
    checks = Tuple{Int,Int}[]
    destinations = Int[]
    for id in copy(p.schedule)
        op = p.operations[id]
        op.active && op.kind isa BroadcastOp && op.readonly || continue
        value = only(op.outputs)
        p.values[value].escapes && continue
        consumers = uses(p, value)
        length(consumers) == 1 && last(only(consumers)) == 1 || continue
        consumer = first(only(consumers))
        next = p.operations[consumer]
        (next.kind isa BroadcastOp || next.kind isa WriteOp{true}) || continue
        rhs = next.kind isa WriteOp ? last(next.expression.args) : next.expression
        readonly_expression(p, rhs) || continue
        guards = delay_checks(p, id, consumer; guarded)
        isnothing(guards) && continue
        before = next.expression
        next.expression = replace_values(v -> v.id == value ? op.expression : v, next.expression)
        next.inputs = operands(next.expression)
        eliminate!(p, op)
        append!(checks, guards)
        push!(destinations, consumer)
        isnothing(on_rewrite) || on_rewrite((;
            before=expression(p, Expr(:block, op.expression, before)),
            fused=expression(p, next.expression)))
    end
    foreach(id -> fuse_destination!(p, id), unique(destinations))
    rebuild!(p)
    return unique(checks)
end

function broadcasted(ex, replacements=Dict{Int,Any}())
    ex isa ValueRef && return get(replacements, ex.id, ex)
    ex isa Expr || return ex
    if _is_broadcast_syntax(ex)
        c = something(_call(ex), _dotcall(ex))
        f = ex.head === :call ? Symbol(chop(string(c.f); head=1, tail=0)) : c.f
        args = map(a -> broadcasted(a, replacements), c.args)
        return Expr(:call, :(Base.broadcasted), f, args...)
    end
    return replace_values(v -> get(replacements, v.id, v), ex)
end

function fused_operation!(p, term, ids; readonly=false)
    outputs = reduce(vcat, [p.operations[i].outputs for i in ids])
    fused = length(p.operations) + 1
    statement = p.operations[last(ids)].statement
    push!(
        p.operations,
        Operation(FusedOp(), term, operands(term), outputs,
            statement, readonly, 0, true),
    )
    foreach(i -> (p.operations[i].active = false), ids)
    foreach(v -> (p.values[v].producer = fused), outputs)
    return fused
end

function multioutput(p)
    q = clone(p)
    ids = [i for i in q.schedule if q.operations[i].active && q.operations[i].kind isa BroadcastOp]
    length(ids) >= 2 || return nothing
    last_broadcast = findlast(i -> q.operations[i].kind isa BroadcastOp, q.schedule)
    allowed(op) =
        op.kind isa Union{BroadcastOp,BindOp,IndexOp,InputOp} ||
        (op.kind isa CallOp && op.readonly && isempty(op.inputs))
    all(enumerate(q.schedule)) do (position, i)
        op = q.operations[i]
        return op.readonly && (allowed(op) ||
                (position > last_broadcast && op.kind isa BorrowingOp))
    end || return nothing
    outputs = [only(q.operations[i].outputs) for i in ids]
    # Moving slices above the combined launch must not read an internal result.
    any(op -> op.kind isa IndexOp && any(in(outputs), op.inputs), q.operations) && return nothing
    replacements = Dict{Int,Any}()
    segments = Any[]
    for (k, id) in enumerate(ids)
        push!(segments, broadcasted(q.operations[id].expression, replacements))
        replacements[outputs[k]] = :(cuNumeric.MatRef($k))
    end
    expression = :(cuNumeric.copyto_fused_multi_alloc!($(Expr(:tuple, segments...))))
    fused = fused_operation!(q, expression, ids; readonly=true)
    hoists = filter(
        i -> !(q.operations[i].kind isa Union{BroadcastOp,BindOp,BorrowingOp}), q.schedule
    )
    binds = filter(i -> q.operations[i].kind isa Union{BindOp,BorrowingOp}, q.schedule)
    q.schedule = [hoists; fused; binds]
    return rebuild!(q)
end

include("graph_siblings.jl")

function optimize(p; fusion=true, aggressive=false, on_rewrite=nothing)
    fusion || return p
    fast = clone(p)
    checks = fuse_producers!(fast; guarded=true, on_rewrite)
    fastplan = aggressive ? fuse_siblings(fast) : fast
    plan = if isempty(checks)
        fastplan
    else
        slow = clone(p)
        fuse_producers!(slow)
        slowplan = aggressive ? fuse_siblings(slow) : slow
        condition = reduce((a, b) -> Expr(:&&, a, b),
            [
                :(cuNumeric._fusion_disjoint($(p.values[a].external), $(p.values[b].external)))
                for (a, b) in checks
            ])
        Choice(Int[], condition, fastplan, slowplan)
    end
    if p.policy isa VisibleScope
        multi = multioutput(p)
        isnothing(multi) || return Choice(Int[], :(cuNumeric._has_gpu_target()), multi, plan)
    end
    return plan
end
