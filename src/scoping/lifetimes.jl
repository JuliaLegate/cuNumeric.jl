# Lifetime analysis when broadcast expressions are evaluated eagerly.

function find_ndarray_assignments(ex, assigned_vars::Set{Symbol})
    cache = Dict{Any,Symbol}()         # expression → temp mapping
    local_assigned = Set{Symbol}()    # track all assigned symbols

    function fresh_tmp(expr)
        counter[] += 1
        tmp = Symbol(:tmp, counter[])
        cache[expr] = tmp
        push!(local_assigned, tmp)
        return tmp, [:($tmp = $expr)]
    end

    function rewrite(e)::Tuple{Any,Vector{Expr}}
        e isa Expr || return e, Expr[]

        if e.head == :(=)
            lhs, rhs = e.args
            lhs isa Symbol && push!(local_assigned, lhs)
            new_rhs, temps = rewrite(rhs)
            return :($lhs = $new_rhs), temps
        end

        if e.head == :(.=)
            lhs, rhs = e.args
            new_lhs, lhs_temps = rewrite(lhs)
            # Do not hoist the top-level call of the RHS to preserve fusion.
            if rhs isa Expr && rhs.head == :call
                op = rhs.args[1]
                new_rhs_args, rhs_temps = Any[], Expr[]
                for arg in rhs.args[2:end]
                    new_arg, temps = rewrite(arg)
                    push!(new_rhs_args, new_arg)
                    append!(rhs_temps, temps)
                end
                new_rhs = Expr(:call, op, new_rhs_args...)
                return Expr(:(.=), new_lhs, new_rhs), vcat(lhs_temps, rhs_temps)
            end

            new_rhs, rhs_temps = rewrite(rhs)
            return Expr(:(.=), new_lhs, new_rhs), vcat(lhs_temps, rhs_temps)
        end

        e.head == :ref && return fresh_tmp(e)

        if e.head == :call
            op = e.args[1]
            new_args, hoisted = Any[], Expr[]
            for arg in e.args[2:end]
                new_arg, temps = rewrite(arg)
                push!(new_args, new_arg)
                append!(hoisted, temps)
            end
            tmp, bind = fresh_tmp(Expr(:call, op, new_args...))
            return tmp, vcat(hoisted, bind)
        end

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
    end
    return Expr(:block, temps..., new_ex)
end

function process_lifetime_scope(scope)
    assigned_vars = Set{Symbol}()
    rewritten = find_ndarray_assignments(scope, assigned_vars)
    result = insert_finalizers(rewritten, assigned_vars)
    counter[] = 0
    return result
end
