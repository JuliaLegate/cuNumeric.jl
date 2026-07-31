# Lifetime analysis for lazy broadcast expression trees.

is_broadcast_op(op) = op isa Symbol && startswith(string(op), ".")

function find_broadcast_assignments(ex, assigned_vars::Set{Symbol})
    local_assigned = Set{Symbol}()

    function fresh_tmp(expr)
        counter[] += 1
        tmp = Symbol(:tmp, counter[])
        push!(local_assigned, tmp)
        return tmp, [:($tmp = $expr)]
    end

    function maphoist(f, args)
        new_args, hoisted = Any[], Expr[]
        for arg in args
            new_arg, temps = f(arg)
            push!(new_args, new_arg)
            append!(hoisted, temps)
        end
        return new_args, hoisted
    end

    # Inside a broadcast tree: hoist slices, keep dotted ops/f.(…) lazy, and
    # delegate anything else to rewrite() (it breaks the tree → real NDArray).
    # The slice cache is scoped to one fused tree so repeated views become one
    # task argument without extending their lifetime across task submissions.
    function rewrite_broadcast(
        expr, slice_cache::Dict{Any,Symbol}
    )::Tuple{Any,Vector{Expr}}
        expr isa Expr || return expr, Expr[]
        if expr.head == :ref
            cached = get(slice_cache, expr, nothing)
            cached === nothing || return cached, Expr[]
            tmp, bind = fresh_tmp(expr)
            slice_cache[expr] = tmp
            return tmp, bind
        end
        if expr.head == :call && is_broadcast_op(expr.args[1])
            args, hoisted = maphoist(
                arg -> rewrite_broadcast(arg, slice_cache), expr.args[2:end]
            )
            return Expr(:call, expr.args[1], args...), hoisted
        end
        if expr.head == :. && length(expr.args) == 2 &&
            expr.args[2] isa Expr && expr.args[2].head == :tuple
            args, hoisted = maphoist(
                arg -> rewrite_broadcast(arg, slice_cache), expr.args[2].args
            )
            return Expr(:., expr.args[1], Expr(:tuple, args...)), hoisted
        end
        return rewrite(expr)
    end

    function rewrite(expr)::Tuple{Any,Vector{Expr}}
        expr isa Expr || return expr, Expr[]
        InterBroadcastFusion.is_log_call(expr) && return expr, Expr[]

        if expr.head == :(=)
            lhs, rhs = expr.args
            lhs isa Symbol && push!(local_assigned, lhs)
            new_rhs, temps = rewrite(rhs)
            return :($lhs = $new_rhs), temps
        end

        # A `.=` RHS is a broadcast tree: only its slices are hoisted.
        if expr.head == :(.=)
            lhs, rhs = expr.args
            # An explicit view preserves indexed `.=` semantics for both Array
            # and NDArray while giving the lifetime pass a wrapper to destroy.
            new_lhs, lhs_temps =
                if lhs isa Expr && lhs.head == :ref
                    view_expr = Base.macroexpand(
                        @__MODULE__,
                        Expr(
                            :macrocall,
                            GlobalRef(Base, Symbol("@view")),
                            LineNumberNode(0),
                            lhs,
                        ),
                    )
                    fresh_tmp(view_expr)
                else
                    rewrite(lhs)
                end
            new_rhs, rhs_temps = rewrite_broadcast(rhs, Dict{Any,Symbol}())
            return Expr(:(.=), new_lhs, new_rhs), vcat(lhs_temps, rhs_temps)
        end

        expr.head == :ref && return fresh_tmp(expr)

        if expr.head == :call && is_broadcast_op(expr.args[1])
            inner, hoisted = rewrite_broadcast(expr, Dict{Any,Symbol}())
            tmp, bind = fresh_tmp(inner)
            return tmp, vcat(hoisted, bind)
        end

        if expr.head == :call
            args, hoisted = maphoist(rewrite, expr.args[2:end])
            tmp, bind = fresh_tmp(Expr(:call, expr.args[1], args...))
            return tmp, vcat(hoisted, bind)
        end

        new_args, hoisted = Any[], Expr[]
        is_block = expr.head == :block || expr.head == :begin
        for arg in expr.args
            new_arg, temps = rewrite(arg)
            if is_block && !(arg isa LineNumberNode)
                append!(new_args, temps)
                push!(new_args, new_arg)
            else
                push!(new_args, new_arg)
                append!(hoisted, temps)
            end
        end
        return Expr(expr.head, new_args...), hoisted
    end

    new_ex, temps = rewrite(ex)
    union!(assigned_vars, local_assigned)
    if new_ex isa Expr && new_ex.head == :block
        return Expr(:block, temps..., new_ex.args...)
    end
    return Expr(:block, temps..., new_ex)
end

function process_broadcast_lifetime_scope(scope)
    assigned_vars = Set{Symbol}()
    scope = InterBroadcastFusion.rewrite_scope(scope)
    rewritten = find_broadcast_assignments(scope, assigned_vars)
    result = insert_finalizers(rewritten, assigned_vars)
    counter[] = 0
    return result
end
