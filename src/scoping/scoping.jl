export @accelerate, @show_lifetimes

include("lifetime_dependencies.jl")

# Syntax normalization precedes the shared graph compiler.
include("util.jl")
using .ScopingUtils
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

    if expr.head === :macrocall && (_is_dot_macro(first(expr.args)) ||
        _base_macro(first(expr.args), caller, Symbol("@views")))
        expanded = Base.macroexpand(caller, expr; recursive=false)
        return _expand_dot_macros(expanded, caller)
    end

    args = map(arg -> _expand_dot_macros(arg, caller), expr.args)
    return Expr(expr.head, args...)
end

function _base_macro(name, caller, expected)
    name isa Symbol && return name === expected && isdefined(caller, name) &&
           getfield(caller, name) === getfield(Base, expected)
    name isa GlobalRef && return name.mod === Base && name.name === expected
    return name == Expr(:., :Base, QuoteNode(expected))
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
        node.head in (:quote, :inert, :let, :function, :->) && return nothing
        assignment = _assignment(node)
        isnothing(assignment) && (assignment = _compound_assignment(node))
        isnothing(assignment) || collect_binding(assignment.lhs)
        foreach(visit, node.args)
        return nothing
    end

    visit(expr)
    return assigned
end

include("graph.jl")
include("graph_runtime.jl")

function process_ndarray_scope(
    scope; on_rewrite=nothing, protected_roots::Set{Symbol}=Set{Symbol}(),
    aggressive::Bool=false,
    caller::Module=@__MODULE__,
    visible::Bool=false,
    local_bindings::Set{Symbol}=protected_roots,
)
    # Broadcast expressions stay lazy only when fusion is enabled; otherwise
    # every call is analyzed as an eager allocation.
    policy = visible ? AccelerateGraph.VisibleScope() : AccelerateGraph.LocalScope()
    return AccelerateGraph.compile(scope; caller, protected=protected_roots,
        locals=union(local_bindings, _assigned_symbols(scope)), policy,
        fusion=FUSE_BROADCAST_EXPRS, aggressive, on_rewrite)
end

# Return the deleted value for a finalizer call.
function _delete_argument(expr)
    call = _call(expr)
    isnothing(call) && return nothing
    if call.f != :(cuNumeric.maybe_insert_delete)
        return nothing
    end
    return only(call.args)
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
            printstyled(io, lpad("✗ free ", 11), deleted, "\n"; color=:red)
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
