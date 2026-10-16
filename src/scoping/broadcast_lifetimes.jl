# Lifetime analysis for lazy broadcast expression trees.
#
#   C[2:end-1, :] .= A[2:end-1, :] .* B[2:end-1, :] .+ 2
#
# hoists only materialized values while leaving the dotted tree intact:
#
#   tmp1 = C[2:end-1, :]
#   tmp2 = A[2:end-1, :]
#   tmp3 = B[2:end-1, :]
#   tmp1 .= tmp2 .* tmp3 .+ 2
#
# The destination and input slices are objects that need lifetime management;
# the `.*` and `.+` nodes are lazy and become one fused broadcast kernel.

function rewrite_broadcast_lifetimes(scope)
    assigned_vars = Set{Symbol}()
    fresh_tmp(expr) = _hoist_temporary(expr, assigned_vars)

    # Inside a broadcast tree: hoist slices, keep dotted ops/f.(…) lazy, and
    # delegate anything else to rewrite_materialized() because it breaks the
    # tree and produces a real NDArray.
    # The slice cache is scoped to one fused tree so repeated views become one
    # task argument without extending their lifetime across task submissions.
    function rewrite_lazy_broadcast(
        expr, slice_cache::Dict{Any,Symbol}
    )::Tuple{Any,Vector{Expr}}
        if !(expr isa Expr)
            return expr, Expr[]
        end
        reference = _reference(expr)
        if !isnothing(reference)
            cached = get(slice_cache, expr, nothing)
            if !isnothing(cached)
                return cached, Expr[]
            end
            tmp, bind = fresh_tmp(expr)
            slice_cache[expr] = tmp
            return tmp, bind
        end
        call = _call(expr)
        if !isnothing(call) && _is_broadcast_op(call.f)
            args, hoisted = _maphoist(
                arg -> rewrite_lazy_broadcast(arg, slice_cache), call.args
            )
            return Expr(:call, call.f, args...), hoisted
        end

        dotcall = _dotcall(expr)
        if !isnothing(dotcall)
            args, hoisted = _maphoist(
                arg -> rewrite_lazy_broadcast(arg, slice_cache), dotcall.args
            )
            return Expr(:., dotcall.f, Expr(:tuple, args...)), hoisted
        end
        return rewrite_materialized(expr)
    end

    function rewrite_materialized(expr)::Tuple{Any,Vector{Expr}}
        if !(expr isa Expr)
            return expr, Expr[]
        end

        # Scalar arithmetic is evaluated while the broadcast tree is built;
        # it does not create an NDArray whose lifetime needs to be tracked.
        _is_scalar_expression(expr) && return expr, Expr[]

        assignment = _assignment(expr)
        if !isnothing(assignment)
            (; lhs, rhs) = assignment
            if lhs isa Symbol
                push!(assigned_vars, lhs)
            end
            new_rhs, temps = rewrite_materialized(rhs)
            return :($lhs = $new_rhs), temps
        end

        # A dotted-assignment RHS is a broadcast tree: only its slices are hoisted.
        broadcast_assignment = _broadcast_assignment(expr)
        if !isnothing(broadcast_assignment)
            (; lhs, rhs) = broadcast_assignment
            op = expr.head
            # NDArray slices are writable views. Hoist the destination slice so
            # the fused broadcast writes through it, then destroy its handle.
            lhs_reference = _reference(lhs)
            if isnothing(lhs_reference)
                new_lhs, lhs_temps = rewrite_materialized(lhs)
            else
                new_lhs, lhs_temps = fresh_tmp(:(Base.@view $lhs))
            end
            new_rhs, rhs_temps = rewrite_lazy_broadcast(rhs, Dict{Any,Symbol}())
            return Expr(op, new_lhs, new_rhs), vcat(lhs_temps, rhs_temps)
        end

        reference = _reference(expr)
        if !isnothing(reference)
            return fresh_tmp(expr)
        end

        call = _call(expr)
        if !isnothing(call) && _is_broadcast_op(call.f)
            inner, hoisted = rewrite_lazy_broadcast(expr, Dict{Any,Symbol}())
            tmp, bind = fresh_tmp(inner)
            return tmp, vcat(hoisted, bind)
        end

        if !isnothing(call)
            args, hoisted = _maphoist(rewrite_materialized, call.args)
            tmp, bind = fresh_tmp(Expr(:call, call.f, args...))
            return tmp, vcat(hoisted, bind)
        end

        return _rewrite_children(rewrite_materialized, expr)
    end

    rewritten, temps = rewrite_materialized(scope)
    return _prepend_statements(rewritten, temps), assigned_vars
end

# Scalars/immutable scalar parameter records cannot share mutable array storage.
_fusion_disjoint(a, b) = isbitstype(typeof(a)) || isbitstype(typeof(b))
_fusion_disjoint(a::AbstractArray, b::AbstractArray) = !Base.mightalias(a, b)
_fusion_disjoint(a::NDArray, b::AbstractArray) = false
_fusion_disjoint(a::AbstractArray, b::NDArray) = false
_fusion_disjoint(a::NDArray, b::NDArray) = !nda_overlaps(a, b)

function process_broadcast_lifetime_scope(
    scope; on_rewrite=nothing, protected_roots=Set{Symbol}(), aggressive::Bool=false
)
    merge = aggressive ? _fuse_sibling_updates : identity
    # Returned producers and caller-owned roots stay materialized: exempt from fusion.
    protected = union(_returned_symbols(scope), protected_roots)
    checks = Tuple{Symbol,Symbol}[]
    guard_roots = setdiff(protected_roots, _assigned_symbols(scope))
    rewritten = InterBroadcastFusion.rewrite_scope(scope;
        on_rewrite, protected, guard_roots, alias_checks=checks)
    fast = merge(
        _process_lifetime_scope(rewritten, rewrite_broadcast_lifetimes; protected_roots)
    )
    isempty(checks) && return fast

    # Analyze each straight-line branch separately, then wrap the complete
    # lifetime-managed bodies. Overlapping inputs retain materialized producers.
    fallback = InterBroadcastFusion.rewrite_scope(scope; protected)
    slow = merge(
        _process_lifetime_scope(fallback, rewrite_broadcast_lifetimes; protected_roots)
    )
    conditions = [:(cuNumeric._fusion_disjoint($a, $b)) for (a, b) in checks]
    condition = reduce((a, b) -> Expr(:&&, a, b), conditions)
    return Expr(:if, condition, fast, slow)
end

# Sibling fusion (`aggressive=true`). After lifetime analysis an update reads
#
#   tmp1 = Base.@view(u_new[2:end-1, 2:end-1])
#   tmp2 = u[2:end-1, 2:end-1]
#   tmp1 .= f(tmp2)
#   cuNumeric.maybe_insert_delete(tmp1); cuNumeric.maybe_insert_delete(tmp2)
#
# Adjacent updates merge into one `copyto_fused_siblings!` call: their slice
# binds move before it and their frees after it. That is safe only for NDArray
# slices, which are views, so other inputs run the original code.

# Array sliced by `tmp = X[...]` or `tmp = @view(X[...])`; all-`:` indexing copies.
function _slice_root(stmt)
    assignment = _assignment(stmt)
    isnothing(assignment) && return nothing
    rhs = MacroTools.isexpr(assignment.rhs, :macrocall) ? last(assignment.rhs.args) : assignment.rhs
    reference = _reference(rhs)
    isnothing(reference) && return nothing
    all(==(:(:)), reference.indices) && return nothing
    return reference.array
end

# Slice binds, then `d .= rhs` (or `name = (d .= rhs)`), then frees.
function _update_group(stmts, i)
    assignment = _assignment(stmts[i])
    result, update = isnothing(assignment) ? (nothing, stmts[i]) : (assignment.lhs, assignment.rhs)
    MacroTools.isexpr(update, :(.=)) && update.args[1] isa Symbol || return nothing
    dest, rhs = update.args
    _is_broadcast_syntax(rhs) || return nothing
    call = _to_broadcasted(rhs, Dict{Symbol,Int}(), Pair{Symbol,Any}[])
    isnothing(call) && return nothing
    first, last = i, i
    while first > 1 && !isnothing(_slice_root(stmts[first - 1]))
        first -= 1
    end
    while last < length(stmts) && !isnothing(_delete_argument(stmts[last + 1]))
        last += 1
    end
    binds = stmts[first:(i - 1)]
    any(b -> _assignment(b).lhs === dest, binds) || return nothing
    return (; first, last, binds, roots=_slice_root.(binds), result, dest, call,
        deletes=stmts[(i + 1):last])
end

function _merge_update_groups(stmts, run)
    dests = [g.dest for g in run]
    calls = [g.call for g in run]
    results = [:($(g.result) = $(g.dest)) for g in run if !isnothing(g.result)]
    merged = Expr(:block,
        (b for g in run for b in g.binds)...,
        :(cuNumeric.copyto_fused_siblings!(($(dests...),), ($(calls...),))),
        results...,
        (d for g in run for d in g.deletes)...,
    )
    original = Expr(:block, stmts[run[1].first:run[end].last]...)
    roots = Base.unique(r for g in run for r in g.roots)
    return Expr(:if, :(cuNumeric._all_ndarrays($(roots...))), merged, original)
end

_all_ndarrays(xs...) = all(x -> x isa NDArray, xs)

function _fuse_sibling_updates(scope)
    stmts = _flatten_statements(scope)
    runs = Vector{Any}[]
    for group in filter(!isnothing, [_update_group(stmts, i) for i in eachindex(stmts)])
        adjacent =
            !isempty(runs) && runs[end][end].last + 1 == group.first &&
            all(g -> g.dest !== group.dest, runs[end])
        adjacent ? push!(runs[end], group) : push!(runs, Any[group])
    end
    filter!(run -> length(run) > 1, runs)
    isempty(runs) && return scope
    out, next = Any[], 1
    for run in runs
        append!(out, stmts[next:(run[1].first - 1)])
        push!(out, _merge_update_groups(stmts, run))
        next = run[end].last + 1
    end
    append!(out, stmts[next:end])
    return Expr(:block, out...)
end
