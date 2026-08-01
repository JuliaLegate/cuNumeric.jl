# Lifetime analysis when broadcast expressions are evaluated eagerly.
#
#   result = f(A[2:end-1, :])
#   consume(result)
#
# becomes a linear sequence of named allocations:
#
#   tmp1 = A[2:end-1, :]
#   tmp2 = f(tmp1)
#   result = tmp2
#   tmp3 = consume(result)
#   tmp3
#
# Finalizer insertion is a separate pass in scoping.jl.

function rewrite_eager_lifetimes(scope)
    assigned_vars = Set{Symbol}()
    fresh_tmp(expr) = _hoist_temporary(expr, assigned_vars)

    function rewrite(expr)::Tuple{Any,Vector{Expr}}
        if !(expr isa Expr)
            return expr, Expr[]
        end

        assignment = _assignment(expr)
        if !isnothing(assignment)
            (; lhs, rhs) = assignment
            if lhs isa Symbol
                push!(assigned_vars, lhs)
            end
            new_rhs, temps = rewrite(rhs)
            return :($lhs = $new_rhs), temps
        end

        broadcast_assignment = _broadcast_assignment(expr)
        if !isnothing(broadcast_assignment)
            (; lhs, rhs) = broadcast_assignment
            new_lhs, lhs_temps = rewrite(lhs)
            # Do not hoist the top-level call of the RHS to preserve fusion.
            call = _call(rhs)
            if !isnothing(call)
                new_rhs_args, rhs_temps = _maphoist(rewrite, call.args)
                new_rhs = Expr(:call, call.f, new_rhs_args...)
                return Expr(:(.=), new_lhs, new_rhs), vcat(lhs_temps, rhs_temps)
            end

            new_rhs, rhs_temps = rewrite(rhs)
            return Expr(:(.=), new_lhs, new_rhs), vcat(lhs_temps, rhs_temps)
        end

        reference = _reference(expr)
        if !isnothing(reference)
            return fresh_tmp(expr)
        end

        call = _call(expr)
        if !isnothing(call)
            new_args, hoisted = _maphoist(rewrite, call.args)
            tmp, bind = fresh_tmp(Expr(:call, call.f, new_args...))
            return tmp, vcat(hoisted, bind)
        end

        return _rewrite_children(rewrite, expr)
    end

    rewritten, temps = rewrite(scope)
    return _prepend_statements(rewritten, temps), assigned_vars
end

function process_lifetime_scope(scope)
    return _process_lifetime_scope(scope, rewrite_eager_lifetimes)
end
