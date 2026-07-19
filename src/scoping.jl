export @analyze_lifetimes, @show_lifetimes

@doc"""
    @analyze_lifetimes expr

Wraps a block of code so that all temporary `NDArray` allocations
(e.g. from slicing or function calls) are tracked and safely freed
at the end of the block. Ensures proper cleanup of GPU memory by
inserting `maybe_insert_delete` calls automatically.

When broadcast fusion is enabled (`FUSE_BROADCAST_EXPRS`), dotted operators
(`.+`, `.*`, etc.) form a lazy `Base.Broadcast.Broadcasted` tree compiled into
a single PTX kernel; intermediate nodes are not real `NDArray` allocations and
are not individually hoisted. The macro automatically selects the
broadcast-aware analysis in that case and the plain analysis otherwise.
"""
macro analyze_lifetimes(block)
    return esc(process_ndarray_scope(block))
end

const counter = Ref(0)

function maybe_insert_delete(var::NDArray)
    return cuNumeric.destroy!(var)
end

maybe_insert_delete(x) = x

"""
    walk_symbols(x) -> Vector{Symbol}

Recursively collect all symbols that appear inside expression `x`.
"""
function walk_symbols(x)
    syms = Symbol[]
    if x isa Symbol
        push!(syms, x)
    elseif x isa Expr
        for a in x.args
            append!(syms, walk_symbols(a))
        end
    elseif x isa AbstractArray
        for a in x
            append!(syms, walk_symbols(a))
        end
    end
    return syms
end

"""
    insert_finalizers(stmts::Vector)
Insert `cuNumeric.maybe_insert_delete(var)` after the last use of each temporary variable.
"""
function insert_finalizers(exprs::Vector, assigned_vars::Set{Symbol})
    uses = Dict{Symbol,Vector{Int}}()
    defs = Dict{Symbol,Int}()
    alias_map = Dict{Symbol,Symbol}()

    # Collect all statements, flattening blocks and skipping LineNumberNodes
    stmts = Any[]
    for expr in exprs
        if expr isa LineNumberNode
            continue
        elseif expr isa Expr && expr.head == :block
            for arg in expr.args
                arg isa LineNumberNode || push!(stmts, arg)
            end
        else
            push!(stmts, expr)
        end
    end

    # Pass 1: collect definitions and uses
    for (i, stmt) in enumerate(stmts)
        stmt isa Expr || continue
        stmt.head == :line && continue

        if stmt.head == :(=)
            lhs, rhs = stmt.args
            if lhs isa Symbol
                defs[lhs] = i
            end
            if lhs isa Symbol && rhs isa Symbol
                alias_map[lhs] = rhs
            end
            for s in walk_symbols(rhs)
                push!(get!(uses, s, Int[]), i)
            end
        else
            for s in walk_symbols(stmt)
                push!(get!(uses, s, Int[]), i)
            end
        end
    end

    for (alias, src) in alias_map
        append!(get!(uses, src, Int[]), get(uses, alias, Int[]))
    end

    # Compute last usage index per variable
    last_use = Dict{Symbol,Int}()
    for (v, idxs) in uses
        last_use[v] = maximum(idxs)
    end

    # `F_u = tmp1` aliases one NDArray under two names; resolve to a canonical rep
    # so it's freed once (double-free is masked only by the ptr=0 null-out).
    function canon(v)
        seen = Set{Symbol}()
        while haskey(alias_map, v) && !(v in seen)
            push!(seen, v)
            v = alias_map[v]
        end
        return v
    end

    is_indexed_assign(s) = s isa Expr && s.head == :(=) && !(s.args[1] isa Symbol)
    result_symbol(s) =
        if s isa Symbol
            s
        else
            (s isa Expr && s.head == :(=) && s.args[1] isa Symbol ? s.args[1] : nothing)
        end

    # Pass 2: insert finalizers
    out = Any[]
    n = length(stmts)
    freed = Set{Symbol}()

    # The block's value escapes to the caller, except for `A[...] = rhs`: Julia
    # returns `rhs` there, but that's a dead temp nobody consumes — free it and
    # return `nothing` rather than leak it or hand back a dangling handle.
    terminal_indexed = n > 0 && is_indexed_assign(stmts[n])

    protected = Set{Symbol}()
    if n > 0 && !terminal_indexed
        rs = result_symbol(stmts[n])
        rs isa Symbol && push!(protected, canon(rs))
    end

    function emit_delete!(v)
        c = canon(v)
        (c in freed || c in protected) && return nothing
        push!(freed, c)
        return push!(out, :(cuNumeric.maybe_insert_delete($v)))
    end

    for (i, stmt) in enumerate(stmts)
        # `v = w` aliases w, so don't finalize w at this statement.
        skip_finalize = Set{Symbol}()
        if stmt isa Expr && stmt.head == :(=)
            lhs, rhs = stmt.args
            if lhs isa Symbol && rhs isa Symbol
                push!(skip_finalize, rhs)
            end
        end

        if i == n && !terminal_indexed
            res_var = Symbol(:res, counter[])
            counter[] += 1
            push!(out, :($res_var = $stmt))
        else
            push!(out, stmt)
        end

        for (v, lasti) in last_use
            if lasti == i && v ∈ assigned_vars && !(v ∈ skip_finalize)
                emit_delete!(v)
            end
        end

        i == n && push!(out, terminal_indexed ? :nothing : res_var)
    end

    return out
end

"""
    insert_finalizers(block::Expr)
Apply finalizer insertion to a `begin ... end` or `:block` expression.
"""
function insert_finalizers(block::Expr, assigned_vars::Set{Symbol})
    if block.head == :block || block.head == :begin
        # Filter out LineNumberNodes before processing
        stmts = [s for s in block.args if !(s isa LineNumberNode)]
        new_stmts = insert_finalizers(stmts, assigned_vars)
        return Expr(:block, new_stmts...)
    else
        error("Expected a begin/block expression")
    end
end

function process_ndarray_scope(block)
    # Broadcast trees need the fusion-aware hoisting; otherwise the
    # plain analysis treats every call (dotted or not) as a real allocation.
    @static if FUSE_BROADCAST_EXPRS
        return process_broadcast_scope(block)
    end
    assigned_vars = Set{Symbol}()
    # Process the entire block at once so lifetimes are tracked across statements
    rewritten = find_ndarray_assignments(block, assigned_vars)
    result = insert_finalizers(rewritten, assigned_vars)
    counter[] = 0
    return result
end

function find_ndarray_assignments(ex, assigned_vars::Set{Symbol})
    cache = Dict{Any,Symbol}()       # expression → temp mapping
    local_assigned = Set{Symbol}()    # track all assigned symbols

    # --- create a fresh temp for any expression ---
    function fresh_tmp(expr)
        counter[] += 1
        tmp = Symbol(:tmp, counter[])
        cache[expr] = tmp
        push!(local_assigned, tmp)
        return tmp, [:($tmp = $expr)]
    end

    # --- recursive rewrite ---
    function rewrite(e)::Tuple{Any,Vector{Expr}}
        if !(e isa Expr)
            return e, Expr[]
        end

        # --- assignment: leave LHS intact ---
        if e.head == :(=)
            lhs, rhs = e.args
            if lhs isa Symbol
                push!(local_assigned, lhs)
            end
            new_rhs, temps = rewrite(rhs)
            return :($lhs = $new_rhs), temps
        end

        # --- broadcasted assignment: preserve fusion ---
        if e.head == :(.=)
            lhs, rhs = e.args
            new_lhs, lhs_temps = rewrite(lhs)
            # Do not hoist the top-level call of the RHS to preserve fusion
            if rhs isa Expr && rhs.head == :call
                op = rhs.args[1]
                new_rhs_args, rhs_temps = Any[], Expr[]
                for arg in rhs.args[2:end]
                    new_arg, t = rewrite(arg)
                    push!(new_rhs_args, new_arg)
                    append!(rhs_temps, t)
                end
                new_rhs = Expr(:call, op, new_rhs_args...)
                return Expr(:(.=), new_lhs, new_rhs), vcat(lhs_temps, rhs_temps)
            else
                new_rhs, rhs_temps = rewrite(rhs)
                return Expr(:(.=), new_lhs, new_rhs), vcat(lhs_temps, rhs_temps)
            end
        end

        # --- array slice reference ---
        if e.head == :ref
            return fresh_tmp(e)
        end

        # --- function calls ---
        if e.head == :call
            op = e.args[1]
            new_args, hoisted = Any[], Expr[]

            # recursively rewrite arguments first
            for arg in e.args[2:end]
                new_arg, temps = rewrite(arg)
                push!(new_args, new_arg)
                append!(hoisted, temps)
            end

            # recreate call with rewritten args
            new_expr = Expr(:call, op, new_args...)

            # always hoist calls (arrays and non-arrays)
            tmp, bind = fresh_tmp(new_expr)
            return tmp, vcat(hoisted, bind)
        end

        # --- fallback for other Expr types ---
        new_args, hoisted = Any[], Expr[]
        is_block = e.head == :block || e.head == :begin
        for arg in e.args
            new_arg, temps = rewrite(arg)
            if is_block && !(arg isa LineNumberNode)
                append!(new_args, temps)
                push!(new_args, new_arg)
            else
                push!(new_args, new_arg)
                append!(hoisted, temps)
            end
        end
        return Expr(e.head, new_args...), hoisted
    end

    new_ex, temps = rewrite(ex)
    union!(assigned_vars, local_assigned)

    if new_ex isa Expr && new_ex.head == :block
        return Expr(:block, temps..., new_ex.args...)
    else
        return Expr(:block, temps..., new_ex)
    end
end

# Broadcast-fusion-aware lifetime analysis. Under fusion, dotted operators
# (.+, .*, …) form a lazy Broadcasted tree lowered to one PTX kernel, so their
# intermediate nodes are not real NDArrays — only slices and the tree root are.
# Non-broadcast sub-expressions break the tree and are hoisted like any call.

is_broadcast_op(op) = op isa Symbol && startswith(string(op), ".")

function find_broadcast_assignments(ex, assigned_vars::Set{Symbol})
    local_assigned = Set{Symbol}()

    function fresh_tmp(expr)
        counter[] += 1
        tmp = Symbol(:tmp, counter[])
        push!(local_assigned, tmp)
        return tmp, [:($tmp = $expr)]
    end

    # Rewrite each arg with `f`, collecting the temps each one hoists.
    function maphoist(f, args)
        new_args, hoisted = Any[], Expr[]
        for a in args
            na, ts = f(a)
            push!(new_args, na)
            append!(hoisted, ts)
        end
        return new_args, hoisted
    end

    # Inside a broadcast tree: hoist slices, keep dotted ops/f.(…) lazy, and
    # delegate anything else to rewrite() (it breaks the tree → real NDArray).
    function rewrite_bcast(e)::Tuple{Any,Vector{Expr}}
        e isa Expr || return e, Expr[]
        e.head == :ref && return fresh_tmp(e)
        if e.head == :call && is_broadcast_op(e.args[1])
            args, hoisted = maphoist(rewrite_bcast, e.args[2:end])
            return Expr(:call, e.args[1], args...), hoisted
        end
        if e.head == :. && length(e.args) == 2 &&
            e.args[2] isa Expr && e.args[2].head == :tuple
            args, hoisted = maphoist(rewrite_bcast, e.args[2].args)
            return Expr(:., e.args[1], Expr(:tuple, args...)), hoisted
        end
        return rewrite(e)
    end

    function rewrite(e)::Tuple{Any,Vector{Expr}}
        e isa Expr || return e, Expr[]

        if e.head == :(=)
            lhs, rhs = e.args
            lhs isa Symbol && push!(local_assigned, lhs)
            new_rhs, temps = rewrite(rhs)
            return :($lhs = $new_rhs), temps
        end

        # .= RHS is a broadcast tree: only the slices inside it are hoisted.
        if e.head == :(.=)
            lhs, rhs = e.args
            new_lhs, lts = rewrite(lhs)
            new_rhs, rts = rewrite_bcast(rhs)
            return Expr(:(.=), new_lhs, new_rhs), vcat(lts, rts)
        end

        e.head == :ref && return fresh_tmp(e)

        # Broadcast root: hoist the fused result as one temp.
        if e.head == :call && is_broadcast_op(e.args[1])
            inner, hoisted = rewrite_bcast(e)
            tmp, bind = fresh_tmp(inner)
            return tmp, vcat(hoisted, bind)
        end

        # Regular call: hoist it and its args.
        if e.head == :call
            args, hoisted = maphoist(rewrite, e.args[2:end])
            tmp, bind = fresh_tmp(Expr(:call, e.args[1], args...))
            return tmp, vcat(hoisted, bind)
        end

        # Other exprs: recurse, splicing hoisted temps inline within blocks.
        new_args, hoisted = Any[], Expr[]
        is_block = e.head == :block || e.head == :begin
        for a in e.args
            na, ts = rewrite(a)
            if is_block && !(a isa LineNumberNode)
                append!(new_args, ts)
                push!(new_args, na)
            else
                push!(new_args, na)
                append!(hoisted, ts)
            end
        end
        return Expr(e.head, new_args...), hoisted
    end

    new_ex, temps = rewrite(ex)
    union!(assigned_vars, local_assigned)
    if new_ex isa Expr && new_ex.head == :block
        return Expr(:block, temps..., new_ex.args...)
    else
        return Expr(:block, temps..., new_ex)
    end
end

function process_broadcast_scope(block)
    assigned_vars = Set{Symbol}()
    rewritten = find_broadcast_assignments(block, assigned_vars)
    result = insert_finalizers(rewritten, assigned_vars)
    counter[] = 0
    return result
end

# Pretty-print the @analyze_lifetimes rewrite (see @show_lifetimes below).
_is_delete_call(s) = Meta.isexpr(s, :call) && s.args[1] == :(cuNumeric.maybe_insert_delete)

# Flatten nested begin/blocks into a linear statement list, dropping line nodes.
function _flatten_stmts(x)
    stmts = Any[]
    function walk(e)
        if Meta.isexpr(e, (:block, :begin))
            foreach(walk, e.args)
        elseif !(e isa LineNumberNode)
            push!(stmts, e)
        end
    end
    walk(x)
    return stmts
end

function print_lifetime_analysis(block; io::IO=stdout)
    rule = "-"^60
    stmts = _flatten_stmts(process_ndarray_scope(block))
    mode = FUSE_BROADCAST_EXPRS ? "fusion-aware" : "plain"

    println(io, "@analyze_lifetimes expansion ($mode analysis)\n", rule)

    n = 0
    for s in stmts
        if _is_delete_call(s)
            printstyled(io, lpad("✗ free ", 11), s.args[2], "\n"; color=:red)
        else
            n += 1
            println(io, lpad(n, 4), "  ", s)
        end
    end

    println(io, rule)
    return nothing
end

@doc"""
    @show_lifetimes expr

Print the lifetime-analysis rewrite of `expr` — the same transformation
[`@analyze_lifetimes`](@ref) applies — without running it. Every statement is
shown in source order and each inserted `maybe_insert_delete` is highlighted so
you can see exactly where each temporary is freed. Pure AST work, so it runs on
CPU-only checkouts.
"""
macro show_lifetimes(block)
    return :(print_lifetime_analysis($(QuoteNode(block))))
end
