using MacroTools: @capture, rmlines, postwalk

export @analyze_lifetimes

"""
    @analyze_lifetimes expr

Wraps a block of code so that all temporary `NDArray` allocations (e.g. from slicing
or function calls) are tracked and safely freed at the end of the block.
Ensures proper cleanup of GPU memory by inserting `maybe_insert_delete` calls.
"""
macro analyze_lifetimes(block)
    esc(process_ndarray_scope(block))
end

function maybe_insert_delete(var::NDArray)
    cuNumeric.nda_destroy_array(var.ptr)
    var.ptr = Ptr{Cvoid}(0)
end
maybe_insert_delete(x) = x

function walk_symbols(x)
    syms = Symbol[]
    postwalk(x) do s
        s isa Symbol && push!(syms, s)
        s
    end
    return syms
end

function process_ndarray_scope(block)
    assigned_vars = Set{Symbol}()
    rewritten = find_ndarray_assignments(block, assigned_vars)
    return insert_finalizers(rewritten, assigned_vars)
end

function insert_finalizers(exprs::Vector, assigned_vars::Set{Symbol})
    stmts = Any[]
    for ex in exprs
        ex isa LineNumberNode && continue
        ex = rmlines(ex)
        # rmlines already strips LineNumberNodes, so args is clean after capture
        if @capture(
            ex,
            begin
                args__
            end
        )
            append!(stmts, args)
        else
            push!(stmts, ex)
        end
    end

    uses = Dict{Symbol,Vector{Int}}()
    alias_map = Dict{Symbol,Symbol}()

    for (i, stmt) in enumerate(stmts)
        if @capture(stmt, lhs_Symbol = rhs_)
            @capture(rhs, src_Symbol) && (alias_map[lhs] = src)
            for s in walk_symbols(rhs)
                push!(get!(uses, s, Int[]), i)
            end
        else
            for s in walk_symbols(stmt)
                push!(get!(uses, s, Int[]), i)
            end
        end
    end

    # If b = a, extend a's lifetime to cover all uses of b
    for (alias, src) in alias_map
        haskey(uses, alias) && append!(get!(uses, src, Int[]), uses[alias])
    end

    last_use = Dict(v => maximum(idxs) for (v, idxs) in uses)

    # Pre-index assigned vars by last-use statement for O(1) per-step lookup
    dying_at = Dict{Int,Vector{Symbol}}()
    for v in assigned_vars
        i = get(last_use, v, 0)
        i > 0 && push!(get!(dying_at, i, Symbol[]), v)
    end

    out = Any[]
    n = length(stmts)
    for (i, stmt) in enumerate(stmts)
        # When stmt is `a = b` (symbol alias), defer freeing b until a's last use
        skip = Set{Symbol}()
        @capture(stmt, _Symbol = rhs_Symbol) && push!(skip, rhs)

        to_free = _vars_to_free(get(dying_at, i, Symbol[]), alias_map, skip)

        if i == n
            res_var = gensym(:res)
            push!(out, :($res_var = $stmt))
            for v in to_free
                # Don't free the value the block itself returns
                is_result = stmt === v || (@capture(stmt, l_ = _) && l === v)
                is_result || push!(out, :(cuNumeric.maybe_insert_delete($v)))
            end
            push!(out, res_var)
        else
            push!(out, stmt)
            for v in to_free
                push!(out, :(cuNumeric.maybe_insert_delete($v)))
            end
        end
    end

    return out
end

# When v = w (alias), free v (the alias holder) rather than w to avoid double-free.
function _vars_to_free(dying, alias_map, skip)
    covered = Set(alias_map[v] for v in dying if haskey(alias_map, v) && alias_map[v] ∈ dying)
    return [v for v in dying if v ∉ skip && v ∉ covered]
end

function insert_finalizers(block::Expr, assigned_vars::Set{Symbol})
    @capture(
        block,
        begin
            stmts__
        end
    ) || error("Expected a begin/block expression")
    Expr(:block, insert_finalizers(Vector{Any}(stmts), assigned_vars)...)
end

function find_ndarray_assignments(ex, assigned_vars::Set{Symbol})
    local_assigned = Set{Symbol}()

    function fresh_tmp(expr)
        tmp = gensym(:tmp)
        push!(local_assigned, tmp)
        return tmp, [:($tmp = $expr)]
    end

    function rewrite(e)
        !(e isa Expr) && return e, Expr[]

        if e.head == :(=)
            lhs, rhs = e.args
            if lhs isa Symbol
                push!(local_assigned, lhs)
                new_rhs, temps = rewrite(rhs)
                return :($lhs = $new_rhs), temps
            elseif lhs isa Expr && lhs.head == :ref
                # Preserve setindex! semantics: pass LHS through, hoist RHS only
                new_rhs, temps = rewrite(rhs)
                return Expr(:(=), lhs, new_rhs), temps
            end
        end

        if e.head == :(.=)
            lhs, rhs = e.args
            new_lhs, lhs_t = rewrite(lhs)
            if rhs isa Expr && rhs.head == :call
                new_args, rhs_t = Any[], Expr[]
                for arg in rhs.args[2:end]
                    na, t = rewrite(arg)
                    push!(new_args, na)
                    append!(rhs_t, t)
                end
                return Expr(:(.=), new_lhs, Expr(:call, rhs.args[1], new_args...)),
                vcat(lhs_t, rhs_t)
            else
                new_rhs, rhs_t = rewrite(rhs)
                return Expr(:(.=), new_lhs, new_rhs), vcat(lhs_t, rhs_t)
            end
        end

        e.head == :ref && return fresh_tmp(e)

        if e.head == :call
            new_args, hoisted = Any[], Expr[]
            for arg in e.args[2:end]
                na, t = rewrite(arg)
                push!(new_args, na)
                append!(hoisted, t)
            end
            tmp, bind = fresh_tmp(Expr(:call, e.args[1], new_args...))
            return tmp, vcat(hoisted, bind)
        end

        new_args, hoisted = Any[], Expr[]
        is_block = e.head == :block || e.head == :begin
        for arg in e.args
            na, t = rewrite(arg)
            if is_block && !(arg isa LineNumberNode)
                append!(new_args, t)
                push!(new_args, na)
            else
                push!(new_args, na)
                append!(hoisted, t)
            end
        end
        return Expr(e.head, new_args...), hoisted
    end

    new_ex, temps = rewrite(ex)
    union!(assigned_vars, local_assigned)

    if @capture(
        new_ex,
        begin
            args__
        end
    )
        return Expr(:block, temps..., args...)
    else
        return Expr(:block, temps..., new_ex)
    end
end
