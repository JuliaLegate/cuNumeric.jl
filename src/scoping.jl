export @analyze_lifetimes

@doc"""
    @analyze_lifetimes expr

Wraps a block of code so that all temporary `NDArray` allocations
(e.g. from slicing or function calls) are tracked and safely freed
at the end of the block. Ensures proper cleanup of GPU memory by
inserting `maybe_insert_delete` calls automatically.
"""
macro analyze_lifetimes(block)
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

    # Pass 2: insert finalizers
    out = Any[]
    n = length(stmts)
    for (i, stmt) in enumerate(stmts)
        # detect aliasing: v = w means don't finalize w
        skip_finalize = Set{Symbol}()
        if stmt isa Expr && stmt.head == :(=)
            lhs, rhs = stmt.args
            if lhs isa Symbol && rhs isa Symbol
                push!(skip_finalize, rhs)
            end
        end

        if i == n
            # Capture result of the last statement
            res_var = Symbol(:res, counter[])
            counter[] += 1
            push!(out, :($res_var = $stmt))

            # Insert finalizers for the last statement
            for (v, lasti) in last_use
                if lasti == i && v ∈ assigned_vars && !(v ∈ skip_finalize)
                    # Do not delete if the result of the block is exactly this variable
                    # or if it's an assignment to this variable.
                    is_result = (stmt === v)
                    if stmt isa Expr && stmt.head == :(=) && stmt.args[1] === v
                        is_result = true
                    end
                    if !is_result
                        push!(out, :(cuNumeric.maybe_insert_delete($v)))
                    end
                end
            end
            # Return the captured result
            push!(out, res_var)
        else
            push!(out, stmt)
            for (v, lasti) in last_use
                if lasti == i && v ∈ assigned_vars && !(v ∈ skip_finalize)
                    push!(out, :(cuNumeric.maybe_insert_delete($v)))
                end
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
