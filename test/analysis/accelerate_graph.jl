using Test
using Graphs

const AG = cuNumeric.AccelerateGraph

function check_graph(p::AG.Program)
    active = filter(id -> p.operations[id].active, p.schedule)
    positions = Dict(id => i for (i, id) in enumerate(active))
    @test !is_cyclic(p.dependencies)
    @test !is_cyclic(p.lifetimes)
    @test all(e -> positions[src(e)] < positions[dst(e)], edges(p.dependencies))
    @test all(active) do id
        all(v -> p.values[v].producer in active, p.operations[id].inputs)
    end
    deleted = reduce(vcat, AG.cleanup(p); init=Int[])
    @test length(deleted) == length(unique(deleted))
end
function check_graph(choice::AG.Choice)
    check_graph(choice.fast)
    return check_graph(choice.slow)
end

function graph_for(ex; protected=Set{Symbol}())
    return AG.parse_scope(ex; caller=@__MODULE__, protected,
        locals=union(protected, cuNumeric._assigned_symbols(ex)))
end

@testset "Graph syntax boundaries" begin
    for body in (:(
            if a
                ;b;
            end
        ), :(
            for x in a
                ;x;
            end
        ), :(a && b), :(x -> x),
        :(function nested(x)
            ;x;
        end))
        outer = Expr(:function, :(outer(a, b)), Expr(:block, body))
        @test_throws ErrorException cuNumeric._accelerate_expand(outer, @__MODULE__)
    end
    quoted = @accelerate quote
        if a
            b = c
        end
    end
    @test quoted isa Expr
    @test quoted.head === :block
    @accelerate function graph_signature((f, x); y=2)
        f(x) + y
    end
    @test graph_signature((abs, -3); y=4) == 7
end

@testset "Graph structure and value identities" begin
    p = graph_for(quote
        x = copy(a)
        old = x
        x = copy(b)
        result = old + old
        (x, result)
    end)
    @test p.bindings[:x] != p.bindings[:old]
    add = only(filter(op -> op.kind isa AG.CallOp && op.expression.args[1] === :+, p.operations))
    @test count(==(p.bindings[:old].id), add.inputs) == 2
    @test !is_cyclic(p.dependencies)
    @test !is_cyclic(p.lifetimes)
    positions = Dict(id => i for (i, id) in enumerate(p.schedule))
    @test all(e -> positions[src(e)] < positions[dst(e)], edges(p.dependencies))
    @test all(
        op -> all(v -> p.values[v].producer == 0 ||
                       p.values[v].producer in p.schedule, op.inputs), p.operations)
    @test AG.clone(p).caller === p.caller
    @test AG.clone(p).values !== p.values
end

@testset "Graph cleanup follows borrowed values" begin
    p = graph_for(quote
        x = copy(a)
        w = transpose(x)
        result = copy(w)
        result
    end)
    x, w = p.bindings[:x].id, p.bindings[:w].id
    @test has_edge(p.lifetimes, x, w)
    deletes = AG.cleanup(p)
    position = only(findall(ids -> x in ids, deletes))
    consumer = p.values[p.bindings[:result].id].producer
    @test position >= findfirst(==(consumer), p.schedule)
    @test !any(ids -> w in ids, deletes)
    @test !any(ids -> p.bindings[:result].id in ids, deletes)
end

@testset "Graph execution preserves reassignments and effects" begin
    @accelerate function graph_reassign(a, b)
        x = copy(a)
        first = x
        x = copy(b)
        second = x
        x = first
        first = second
        (copy(x), copy(first))
    end
    @test graph_reassign([1, 2], [3, 4]) == ([1, 2], [3, 4])
    @accelerate function graph_reassign_input(a, b)
        old = a
        a = b
        copy(old)
    end
    @test graph_reassign_input([1, 2], [3, 4]) == [1, 2]
    @accelerate function graph_repeat(a)
        x = a .+ 1
        x .* x
    end
    @test graph_repeat([1, 2]) == [4, 9]
    @accelerate function graph_indexed_result(a, dest)
        x = a .+ 1
        y = x .* 2
        dest[:] = y
        nothing
    end
    dest = zeros(Int, 2)
    graph_indexed_result([1, 2], dest)
    @test dest == [4, 6]
    @accelerate function graph_delayed_indexed_result(a, dest)
        x = a .+ 1
        y = x .* 2
        fill!(a, 10)
        dest[:] = y
        nothing
    end
    graph_delayed_indexed_result([1, 2], dest)
    @test dest == [4, 6]
    @accelerate function graph_shadow(a, f)
        result = f(a)
        f = identity
        f(result)
    end
    @test graph_shadow([1, 2], reverse) == [2, 1]
    @accelerate function graph_mutation(a, b)
        x = a .+ 1
        b[1] = 9
        x .* 2
    end
    a = [1, 2]
    @test graph_mutation(a, a) == [4, 6]
    @test a == [9, 2]
    @accelerate function graph_nested_assignment(a)
        y = (x = copy(a))
        x = copy(y)
        (x, y)
    end
    @test graph_nested_assignment([1, 2]) == ([1, 2], [1, 2])
    @accelerate function graph_scope(a)
        x = copy(a)
        (@isdefined(x), x)
    end
    @test graph_scope([1, 2]) == (true, [1, 2])
    @accelerate function graph_typed_assignment(a)
        x::Int = a
        x = 2.0
        x
    end
    @test (@inferred graph_typed_assignment(1.0)) === 2
    function graph_let_headers(x)
        result = @accelerate let x = copy(x), y = x
            x = copy(y)
            (x, y)
        end
        return result, x
    end
    @test graph_let_headers([1, 2]) == (([1, 2], [1, 2]), [1, 2])
    function graph_let_body_shadow(x)
        @accelerate let y = copy(x)
            x = copy(y)
            x
        end
    end
    @test graph_let_body_shadow([3, 4]) == [3, 4]
    @accelerate function graph_views(a)
        @views x = a[2:end] .+ 1
        old = x
        @views x = a[1:(end - 1)] .+ 2
        (old, x)
    end
    @test graph_views([1, 3, 5]) == ([4, 6], [3, 5])
    record!(events, label, value) = (push!(events, label); value)
    with_keyword(x; y) = x + y
    @accelerate function graph_keywords(events)
        with_keyword(record!(events, :positional, 1); y=record!(events, :keyword, 2))
    end
    events = Symbol[]
    @test graph_keywords(events) == 3
    @test events == [:positional, :keyword]
    @accelerate function graph_callable_order(events)
        record!(events, :function, abs).(record!(events, :argument, [-1, 2]))
    end
    empty!(events)
    @test graph_callable_order(events) == [1, 2]
    @test events == [:function, :argument]
    @accelerate function graph_callable_rebinding(f, a)
        f.((f=identity; a))
    end
    @test graph_callable_rebinding(abs, [-1, 2]) == [1, 2]
    @accelerate function graph_index_effects(a, events)
        a[record!(events, :index, 1:2)] .+ a[record!(events, :index, 1:2)]
    end
    empty!(events)
    @test graph_index_effects([1, 2], events) == [2, 4]
    @test events == [:index, :index]
    @accelerate function graph_slice_rebinding(a, b)
        a[1:2] .+ (a = b) .+ a[1:2]
    end
    @test graph_slice_rebinding([1, 2], [3, 4]) == [7, 10]
    @accelerate function graph_slice_mutation(a)
        a[1:2] .+ (a[1] = 10) .+ a[1:2]
    end
    @test graph_slice_mutation([1, 2]) == [21, 14]
    @accelerate function graph_index_rebinding(a)
        i = 1
        x = a[(i = 2):end]
        (x, i)
    end
    @test graph_index_rebinding([1, 2, 3]) == ([2, 3], 2)
    @accelerate function graph_destructure(a)
        x, ys... = a
        (x, ys)
    end
    @test graph_destructure((1, 2, 3)) == (1, (2, 3))
    @accelerate function graph_compound(x)
        old = x
        x += 2
        (old, x)
    end
    @test graph_compound(3) == (3, 5)
end

@testset "Graph producer and sibling fusion" begin
    source = quote
        x = a .+ 1
        y = x .* 2
        (x, y)
    end
    multi = AG.parse_scope(source; caller=@__MODULE__,
        locals=Set([:x, :y]), protected=Set([:x, :y]), policy=AG.VisibleScope())
    fused_multi = AG.multioutput(multi)
    @test !isnothing(fused_multi)
    @test occursin("copyto_fused_multi_alloc!", string(AG.emit(fused_multi)))
    for source in (quote
            x = touch.(a)
            y = b .+ 1
        end, quote
            x = a .+ 1
            y = b[touch(a)] .+ 1
        end)
        @test isnothing(AG.multioutput(graph_for(source)))
    end
    p = graph_for(quote
        t = a .+ 1
        u = t .* 2
        u .+ 3
    end)
    checks = AG.fuse_producers!(p)
    @test isempty(checks)
    @test count(op -> op.active && op.kind isa AG.BroadcastOp, p.operations) == 1
    @test !is_cyclic(p.dependencies)
    check_graph(fused_multi)
    indexed = graph_for(quote
            t = a[lo:hi] .+ 1
            b[lo:hi] .= c[lo:hi]
            t .* 2
        end; protected=Set([:a, :b, :c, :lo, :hi]))
    reads = filter(op -> op.kind isa AG.IndexOp, indexed.operations)
    @test AG.storage_roots(indexed, only(first(reads).outputs)) == [indexed.bindings[:a].id]
    guards = AG.fuse_producers!(indexed; guarded=true)
    @test guards == [(indexed.bindings[:b].id, indexed.bindings[:a].id)]
    check_graph(indexed)
    check_graph(AG.optimize(graph_for(quote
        x = a .+ 1
        y = x .* 2
        dest[:] = y
        nothing
    end)))
    check_graph(
        AG.optimize(graph_for(quote
                    t = a .+ 1
                    b[1:4] .= c[1:4]
                    t .* 2
                end; protected=Set([:a, :b, :c])); aggressive=true),
    )
    check_graph(
        AG.optimize(graph_for(quote
                    x[1:4] .= a[1:4] .+ 1
                    y[1:4] .= x[1:2:7] .* 2
                    z[1:4] .= y[1:4] .+ 3
                end; protected=Set([:a, :x, :y, :z])); aggressive=true),
    )

    @accelerate aggressive=true function graph_siblings(a, x, y, z)
        x[1:4] .= a[1:4] .+ 1
        y[1:4] .= a[1:4] .* x[1]
        z[1:4] .= x[1:2:7] .* 2
    end
    function reference_siblings(a, x, y, z)
        x[1:4] .= a[1:4] .+ 1
        y[1:4] .= a[1:4] .* x[1]
        z[1:4] .= x[1:2:7] .* 2
    end
    for overlap in (false, true)
        a = collect(1.0:8.0)
        x, y, z = copy(a), copy(a), copy(a)
        overlap && (a = x)
        ar, xr, yr, zr = copy(a), copy(x), copy(y), copy(z)
        overlap && (ar = xr)
        graph_siblings(a, x, y, z)
        reference_siblings(ar, xr, yr, zr)
        @test (x, y, z) == (xr, yr, zr)
    end
end
