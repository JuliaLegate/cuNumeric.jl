module ScopingUtils

using MacroTools: MacroTools

# Shared syntax vocabulary for every scoping pass. For example:
#
#   _assignment(:(result = A .+ B))
#       -> (lhs=:result, rhs=:(A .+ B))
# Keeping these matches here means the transformation files describe policy
# instead of repeating Expr head/argument indexing.

export _assignment, _broadcast_assignment, _compound_assignment, _call, _dotcall,
    _flatten_statements, _is_broadcast_op, _is_broadcast_syntax,
    _is_scalar_expression, _scope_statements, _strip_lines,
    walk_symbols

function _assignment(expr)
    MacroTools.isexpr(expr, :(=)) || return nothing
    MacroTools.@capture(expr, lhs_ = rhs_) || return nothing
    return (; lhs, rhs)
end

function _broadcast_assignment(expr)
    expr isa Expr && length(expr.args) == 2 || return nothing
    op = expr.head
    op isa Symbol || return nothing
    spelling = string(op)
    startswith(spelling, ".") && endswith(spelling, "=") || return nothing
    return (; lhs=expr.args[1], rhs=expr.args[2])
end

function _compound_assignment(expr)
    expr isa Expr && length(expr.args) == 2 || return nothing
    spelling = string(expr.head)
    startswith(spelling, ".") && return nothing
    endswith(spelling, "=") || return nothing
    spelling in ("=", "==", "!=", "<=", ">=", "===", "!==") && return nothing
    return (; lhs=first(expr.args), rhs=last(expr.args),
        op=Symbol(chop(spelling; head=0, tail=1)))
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

const _SCALAR_ARITHMETIC = (:+, :-, :*, :/, :^, :%, :fld, :cld, :mod, :rem)

function _is_property_access(expr)
    return MacroTools.isexpr(expr, :.) && length(expr.args) == 2 &&
           expr.args[2] isa QuoteNode
end

_is_scalar_expression(::Number) = true
_is_scalar_expression(expr::QuoteNode) = _is_scalar_expression(expr.value)
_is_scalar_expression(::Any) = false

function _is_scalar_expression(expr::Expr)
    _is_property_access(expr) && return true

    call = _call(expr)
    return !isnothing(call) && call.f in _SCALAR_ARITHMETIC &&
           all(_is_scalar_expression, call.args)
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
