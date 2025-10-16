# TODO reduce number of allocations. Potentially remove assigned_vars somehow

export @cunumeric

@doc"""
    @cunumeric expr

Wraps a block of code so that all temporary `NDArray` allocations 
(e.g. from slicing or function calls) are tracked and safely freed 
at the end of the block. Ensures proper cleanup of GPU memory by 
inserting `maybe_insert_delete` calls automatically.
"""
macro cunumeric(block)
    esc(process_ndarray_scope(block))
end

const counter = Ref(0)

function maybe_insert_delete(var::NDArray)
    cuNumeric.nda_destroy_array(var.ptr)
    var.ptr = Ptr{Cvoid}(0)
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

    stmts = Any[]
    for expr in exprs
        append!(stmts, expr.args)
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

    # Pass 2: insert finalizers
    out = Any[]
    for (i, stmt) in enumerate(stmts)
        push!(out, stmt)
        stmt isa Expr || continue
        stmt.head == :line && continue

        # detect aliasing: v = w means don't finalize w
        skip_finalize = Set{Symbol}()
        if stmt.head == :(=)
            lhs, rhs = stmt.args
            # a = tmp1
            # tmp1 will be added to skip_finalize
            # a[:,:] = tmp1
            # this does a copy, so we want to finalize tmp1
            if lhs isa Symbol && rhs isa Symbol
                push!(skip_finalize, rhs)
            end
        end

        for (v, lasti) in last_use
            if lasti == i && v ∈ assigned_vars && !(v ∈ skip_finalize)
                push!(out, :(cuNumeric.maybe_insert_delete($v)))
            end
        end
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
    # Normalize block to list of statements
    stmts = block isa Expr && block.head == :block ? block.args : [block]

    # Otherwise, process and cache
    assigned_vars = Set{Symbol}()
    body = Any[]

    for stmt in stmts
        stmts = find_ndarray_assignments(stmt, assigned_vars)
        new_stmts = insert_finalizers(stmts, assigned_vars)
        push!(body, new_stmts)
    end

    println(body)

    result = quote
        $(Expr(:block, body...))
    end

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
            return Expr(:block, temps..., :($lhs = $new_rhs)), []
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
        for arg in e.args
            new_arg, temps = rewrite(arg)
            push!(new_args, new_arg)
            append!(hoisted, temps)
        end
        return Expr(e.head, new_args...), hoisted
    end

    new_ex, temps = rewrite(ex)
    union!(assigned_vars, local_assigned)
    return Expr(:block, temps..., new_ex)
end
