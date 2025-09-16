export @cunumeric

macro cunumeric(block)
    esc(process_ndarray_scope(block))
end

const ndarray_scope_cache = Dict{UInt64,Expr}()
const counter = Ref(0)

function maybe_insert_delete(var::NDArray)
    cuNumeric.nda_destroy_array(var.ptr)
    var.ptr = Ptr{Cvoid}(0)
end

maybe_insert_delete(x) = x

function process_ndarray_scope(block)
    # Normalize block to list of statements
    stmts = block isa Expr && block.head == :block ? block.args : [block]

    # Hash the content structurally (avoiding pointer identity)
    h = hash(stmts)

    # Return cached result if present
    if haskey(ndarray_scope_cache, h)
        return ndarray_scope_cache[h]
    end

    # Otherwise, process and cache
    assigned_vars = Set{Symbol}()
    body = Any[]

    for stmt in stmts
        stmt = find_ndarray_assignments(stmt, assigned_vars)
        push!(body, stmt)
    end

    cleanup_exprs = []

    for var in assigned_vars
        push!(cleanup_exprs, quote
            cuNumeric.maybe_insert_delete($var)
        end)
    end

    result = quote
        cuNumeric.@__tryfinally($(Expr(:block, body...)),
            $(Expr(:block, cleanup_exprs...)))
    end

    counter[] = 0
    ndarray_scope_cache[h] = result
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
