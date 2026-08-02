# Lifetime analysis for lazy broadcast expression trees.
#
#   C[2:end-1, :] .= A[2:end-1, :] .* B[2:end-1, :] .+ 2
#
# hoists only materialized values while leaving the dotted tree intact:
#
#   tmp1 = @view C[2:end-1, :]
#   tmp2 = A[2:end-1, :]
#   tmp3 = B[2:end-1, :]
#   tmp1 .= tmp2 .* tmp3 .+ 2
#
# The destination view and input slices are objects that need lifetime management; 
# the `.*` and `.+` nodes are lazy and become one fused broadcast kernel.

function _expand_view(expr)
    return Base.macroexpand(@__MODULE__, :(@view $expr))
end

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

        assignment = _assignment(expr)
        if !isnothing(assignment)
            (; lhs, rhs) = assignment
            if lhs isa Symbol
                push!(assigned_vars, lhs)
            end
            new_rhs, temps = rewrite_materialized(rhs)
            return :($lhs = $new_rhs), temps
        end

        # A `.=` RHS is a broadcast tree: only its slices are hoisted.
        broadcast_assignment = _broadcast_assignment(expr)
        if !isnothing(broadcast_assignment)
            (; lhs, rhs) = broadcast_assignment
            # An explicit view preserves indexed `.=` semantics for both Array
            # and NDArray while giving the lifetime pass a wrapper to destroy.
            lhs_reference = _reference(lhs)
            if isnothing(lhs_reference)
                new_lhs, lhs_temps = rewrite_materialized(lhs)
            else
                new_lhs, lhs_temps = fresh_tmp(_expand_view(lhs))
            end
            new_rhs, rhs_temps = rewrite_lazy_broadcast(rhs, Dict{Any,Symbol}())
            return Expr(:(.=), new_lhs, new_rhs), vcat(lhs_temps, rhs_temps)
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

function process_broadcast_lifetime_scope(scope; on_rewrite=nothing)
    scope = InterBroadcastFusion.rewrite_scope(scope; on_rewrite)
    return _process_lifetime_scope(scope, rewrite_broadcast_lifetimes)
end
