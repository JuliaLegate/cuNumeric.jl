module ScopingUtils

using MacroTools: MacroTools

# Shared syntax vocabulary for every scoping pass. For example:
#
#   _assignment(:(result = A .+ B))
#       -> (lhs=:result, rhs=:(A .+ B))
#   _reference(:(A[2:end-1, :]))
#       -> (array=:A, indices=Any[:(2:end-1), :(:)])
#
# Keeping these matches here means the transformation files describe policy
# instead of repeating Expr head/argument indexing.

export _assignment, _broadcast_assignment, _call, _dotcall,
    _flatten_statements, _is_broadcast_op, _is_broadcast_syntax, _maphoist,
    _reference, _prepend_statements, _replace_symbols, _rewrite_children,
    _scope_statements, _strip_lines, walk_symbols

function _assignment(expr)
    MacroTools.isexpr(expr, :(=)) || return nothing
    MacroTools.@capture(expr, lhs_ = rhs_) || return nothing
    return (; lhs, rhs)
end

function _broadcast_assignment(expr)
    MacroTools.isexpr(expr, :(.=)) || return nothing
    MacroTools.@capture(expr, lhs_ .= rhs_) || return nothing
    return (; lhs, rhs)
end

function _call(expr)
    MacroTools.isexpr(expr, :call) || return nothing
    MacroTools.@capture(expr, f_(args__)) || return nothing
    return (; f, args)
end

function _dotcall(expr)
    MacroTools.isexpr(expr, :.) || return nothing
    MacroTools.@capture(expr, f_.(args__)) || return nothing
    return (; f, args)
end

_is_broadcast_op(op) = op isa Symbol && startswith(string(op), ".")

function _is_broadcast_syntax(expr)
    call = _call(expr)
    if !isnothing(call) && _is_broadcast_op(call.f)
        return true
    end
    return !isnothing(_dotcall(expr))
end

function _reference(expr)
    MacroTools.isexpr(expr, :ref) || return nothing
    MacroTools.@capture(expr, array_[indices__]) || return nothing
    return (; array, indices)
end

_is_line(expr) = MacroTools.isline(expr)
_is_scope(expr) = MacroTools.isexpr(expr, :block, :begin)
_strip_lines(expr) = MacroTools.striplines(expr)

function _scope_statements(scope)
    _is_scope(scope) || return nothing
    return MacroTools.rmlines(scope).args
end

function _flatten_statements(scope)
    flattened = MacroTools.flatten(scope)
    _is_line(flattened) && return Any[]
    statements = _scope_statements(flattened)
    isnothing(statements) && return Any[flattened]
    return Any[statements...]
end

function _maphoist(transform, expressions)
    rewritten = Any[]
    hoisted = Expr[]
    for expr in expressions
        new_expr, temps = transform(expr)
        push!(rewritten, new_expr)
        append!(hoisted, temps)
    end
    return rewritten, hoisted
end

function _rewrite_children(transform, expr::Expr)
    rewritten = Any[]
    hoisted = Expr[]
    is_scope = _is_scope(expr)
    for arg in expr.args
        new_arg, temps = transform(arg)
        if is_scope && !_is_line(arg)
            append!(rewritten, temps)
            push!(rewritten, new_arg)
        else
            push!(rewritten, new_arg)
            append!(hoisted, temps)
        end
    end
    return Expr(expr.head, rewritten...), hoisted
end

function _prepend_statements(expr, statements)
    if _is_scope(expr)
        return Expr(:block, statements..., expr.args...)
    end
    return Expr(:block, statements..., expr)
end

function _replace_symbols(expr, replacements::AbstractDict{Symbol})
    return MacroTools.postwalk(expr) do node
        node isa Symbol || return node
        return get(replacements, node, node)
    end
end

"""
    walk_symbols(x) -> Vector{Symbol}

Recursively collect all symbols that appear inside expression `x`.
"""
function walk_symbols(x)
    syms = Symbol[]
    MacroTools.postwalk(x) do node
        node isa Symbol && push!(syms, node)
        if node isa AbstractArray
            for element in node
                append!(syms, walk_symbols(element))
            end
        end
        return node
    end
    return syms
end

end
