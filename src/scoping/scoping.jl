export @accelerate, @show_lifetimes

include("lifetime_dependencies.jl")
include("lifetime_runtime.jl")

# Include generic syntax layers before the cuNumeric-specific lifetime passes.
include("util.jl")
using .ScopingUtils

include("inter_broadcast_fusion.jl")

const _DOT_MACRO_NAME = Symbol("@__dot__")

_is_dot_macro(::Any) = false
_is_dot_macro(name::Symbol) = name === _DOT_MACRO_NAME
_is_dot_macro(name::GlobalRef) = name.name === _DOT_MACRO_NAME

function _is_dot_macro(name::Expr)
    name.head === :. || return false
    quoted_name = last(name.args)
    return quoted_name isa QuoteNode && quoted_name.value === _DOT_MACRO_NAME
end

_expand_dot_macros(value, ::Module) = value

function _expand_dot_macros(expr::Expr, caller::Module)
    expr.head in (:quote, :inert) && return expr

    if expr.head === :macrocall && _is_dot_macro(first(expr.args))
        expanded = Base.macroexpand(caller, expr; recursive=false)
        return _expand_dot_macros(expanded, caller)
    end

    args = map(arg -> _expand_dot_macros(arg, caller), expr.args)
    return Expr(expr.head, args...)
end

# A block yields its final expression, not a `return` (which exits the enclosing
# function). Accept `return expr` only as the final statement and rewrite it to a
# plain trailing expression so it flows through analysis like any other value.
function _normalize_return(block)
    if block isa Expr && block.head === :return
        return isempty(block.args) ? :nothing : only(block.args)
    end
    stmts = _scope_statements(block)
    isnothing(stmts) && return block
    for (i, stmt) in enumerate(stmts)
        stmt isa Expr && stmt.head === :return || continue
        i == length(stmts) || throw(
            ArgumentError("`return` is only allowed as the final statement")
        )
        value = isempty(stmt.args) ? :nothing : only(stmt.args)
        return Expr(block.head, stmts[1:(end - 1)]..., value)
    end
    return block
end

const counter = Ref(0)

function maybe_insert_delete(var::NDArray)
    return cuNumeric.destroy!(var)
end

maybe_insert_delete(x) = x

# Symbols bound by an assignment anywhere in `expr` (let-form locals; soft-form
# protected bindings).
function _assigned_symbols(expr)
    assigned = Set{Symbol}()

    function collect_binding(lhs)
        if lhs isa Symbol
            push!(assigned, lhs)
        elseif lhs isa Expr && lhs.head in (:tuple, :parameters)
            foreach(collect_binding, lhs.args)
        elseif lhs isa Expr && lhs.head in (:(::), :(...))
            collect_binding(first(lhs.args))
        end
        return nothing
    end

    function visit(node)
        node isa Expr || return nothing
        assignment = _assignment(node)
        isnothing(assignment) || collect_binding(assignment.lhs)
        foreach(visit, node.args)
        return nothing
    end

    visit(expr)
    return assigned
end

function _lexical_scope(body, bindings::Set{Symbol})
    ordered = Base.sort!(collect(bindings); by=string)
    return Expr(:let, Expr(:block, ordered...), body)
end

function _hoist_temporary(expr, assigned_vars)
    counter[] += 1
    temporary = Symbol(:tmp, counter[])
    push!(assigned_vars, temporary)
    return temporary, [:($temporary = $expr)]
end

# Symbols a statement yields directly, used to keep returned producers out of
# broadcast fusion. Actual escaping handle dependencies are protected at runtime.
function _result_symbols(stmt)
    stmt isa Symbol && return Set([stmt])

    assignment = _assignment(stmt)
    if !isnothing(assignment)
        return _result_symbols(assignment.rhs)
    end

    if stmt isa Expr && stmt.head in (:tuple, :parameters)
        return mapreduce(_result_symbols, union, stmt.args; init=Set{Symbol}())
    end
    if stmt isa Expr && stmt.head == :kw
        return _result_symbols(last(stmt.args))
    end
    if stmt isa Expr && stmt.head == :(::)
        return _result_symbols(first(stmt.args))
    end

    return Set{Symbol}()
end

function _returned_symbols(scope)
    stmts = _scope_statements(scope)
    (isnothing(stmts) || isempty(stmts)) && return Set{Symbol}()
    return _result_symbols(last(stmts))
end

# Syntax determines when a binding is last used; runtime dispatch determines
# which handles its value keeps alive. No constructor names are special here.
function _lifetime_bindings(stmt)
    assignment = _assignment(stmt)
    isnothing(assignment) && return Symbol[]
    names = _assigned_symbols(Expr(:(=), assignment.lhs, nothing))
    filter!(name -> !all(==('_'), string(name)), names)
    return Base.sort!(collect(names); by=string)
end

function insert_finalizers(
    exprs::Vector, ::Set{Symbol}; protected_roots::Set{Symbol}=Set{Symbol}()
)
    stmts = _flatten_statements(Expr(:block, exprs...))
    isempty(stmts) && return stmts
    bindings = map(_lifetime_bindings, stmts)
    assigned = union(Set{Symbol}(), bindings...)
    isempty(assigned) && return stmts
    last_use = Dict{Symbol,Int}()
    used = Set{Symbol}()
    for (i, stmt) in enumerate(stmts), name in walk_symbols(stmt)
        push!(used, name)
        name in assigned && (last_use[name] = i)
    end

    scope, result = gensym(:lifetimes), gensym(:result)
    out = Any[:($scope = cuNumeric._LifetimeScope())]
    # External values may be captured variables as well as function arguments.
    # Soft-scope named bindings also protect any value they had on entry.
    incoming = union(setdiff(used, assigned), protected_roots)
    for name in Base.sort!(collect(incoming); by=string)
        all(==('_'), string(name)) && continue
        push!(out, Expr(:if, Expr(:isdefined, name),
                        :(cuNumeric._lifetime_protect!($scope, $name))))
    end

    # Preserve the existing convention that a terminal indexed write yields
    # nothing, so its otherwise-unused RHS can be released.
    terminal = something(_assignment(last(stmts)),
                         _broadcast_assignment(last(stmts)), (; lhs=nothing))
    terminal_indexed = terminal.lhs isa Expr && terminal.lhs.head === :ref
    for (i, stmt) in enumerate(stmts)
        final = i == length(stmts)
        push!(out, final && !terminal_indexed ? :($result = $stmt) : stmt)
        names = bindings[i]
        if !isempty(names)
            keys = Expr(:tuple, QuoteNode.(names)...)
            values = Expr(:tuple, names...)
            protected = Expr(:tuple, (name in protected_roots for name in names)...)
            push!(out, :(cuNumeric._lifetime_bind!($scope, $keys, $values, $protected)))
        end
        if final && !terminal_indexed
            push!(out, :(cuNumeric._lifetime_protect!($scope, $result)))
        end
        for name in Base.sort!([name for (name, last) in last_use if last == i]; by=string)
            push!(out, :(cuNumeric.maybe_insert_delete($scope, $(QuoteNode(name)))))
        end
    end
    push!(out, :(cuNumeric._lifetime_finish!($scope)))
    push!(out, terminal_indexed ? :nothing : result)
    return out
end

"""
    insert_finalizers(block::Expr)
Apply finalizer insertion to a `begin ... end` or `:block` expression.
"""
function insert_finalizers(
    block::Expr, assigned_vars::Set{Symbol}; protected_roots::Set{Symbol}=Set{Symbol}()
)
    stmts = _scope_statements(block)
    isnothing(stmts) && error("Expected a begin/block expression")
    return Expr(:block, insert_finalizers(stmts, assigned_vars; protected_roots)...)
end

function _process_lifetime_scope(
    scope, rewrite_lifetimes; protected_roots::Set{Symbol}=Set{Symbol}()
)
    try
        rewritten, assigned_vars = rewrite_lifetimes(scope)
        return insert_finalizers(rewritten, assigned_vars; protected_roots)
    finally
        counter[] = 0
    end
end

# Package-specific passes. The broadcast-aware pass also consumes the generic
# inter-broadcast fusion module included above.
include("lifetimes.jl")
include("broadcast_lifetimes.jl")

function process_ndarray_scope(
    scope; on_rewrite=nothing, protected_roots::Set{Symbol}=Set{Symbol}()
)
    # Broadcast expressions stay lazy only when fusion is enabled; otherwise
    # every call is analyzed as an eager allocation.
    @static if FUSE_BROADCAST_EXPRS
        return process_broadcast_lifetime_scope(scope; on_rewrite, protected_roots)
    end
    return process_lifetime_scope(scope; protected_roots)
end

# Return the released binding (or legacy deleted value) for a finalizer call.
function _delete_argument(expr)
    call = _call(expr)
    isnothing(call) && return nothing
    if call.f != :(cuNumeric.maybe_insert_delete)
        return nothing
    end
    return length(call.args) == 1 ? only(call.args) : last(call.args).value
end

# Header + body statements per form, so the printout mirrors the real expansion.
function _analysis_parts(ex)
    ex = _strip_lines(ex)
    if ex isa Expr && ex.head === :function
        return "function " * string(first(ex.args)), _flatten_statements(ex.args[2])
    elseif ex isa Expr && ex.head === :let
        binds = _strip_lines(first(ex.args))
        bindstr = binds isa Expr ? join(binds.args, ", ") : string(binds)
        return "let " * bindstr, _flatten_statements(ex.args[2])
    end
    return nothing, _flatten_statements(ex)
end

# Pretty-print `expansion` (the exact `@accelerate` output), highlighting frees.
function print_lifetime_analysis(expansion; io::IO=stdout)
    rule = "-"^60
    header, stmts = _analysis_parts(expansion)
    mode = FUSE_BROADCAST_EXPRS ? "fusion-aware" : "plain"

    println(io, "@accelerate expansion ($mode)\n", rule)
    isnothing(header) || println(io, header)

    n = 0
    for s in stmts
        deleted = _delete_argument(s)
        if !isnothing(deleted)
            printstyled(io, lpad("✗ release ", 11), deleted, "\n"; color=:red)
        else
            n += 1
            println(io, lpad(n, 4), "  ", _strip_lines(s))
        end
    end

    isnothing(header) || println(io, "end")
    println(io, rule)
    return nothing
end

include("accelerate.jl")
