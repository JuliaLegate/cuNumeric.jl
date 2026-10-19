# Parsing is the only pass that interprets user syntax. Later passes operate on
# value references and operation kinds, including inside lazy broadcast trees.
function constant_binding(mod, name)
    return isdefined(mod, name) && isconst(mod, name) ? getfield(mod, name) : nothing
end
callee(p, ex) = nothing
callee(p, ex::Symbol) = ex in p.locals ? nothing : constant_binding(p.caller, ex)
callee(p, ex::GlobalRef) = constant_binding(ex.mod, ex.name)
function callee(p, ex::Expr)
    ex.head === :. && last(ex.args) isa QuoteNode || return nothing
    mod, name = callee(p, first(ex.args)), last(ex.args).value
    return mod isa Module && name isa Symbol ? constant_binding(mod, name) : nothing
end

const READONLY = Set((:+, :-, :*, :/, :\, :^, :%, :fld, :cld, :mod, :rem, :(:),
    :abs, :abs2, :sqrt, :exp, :log, :sin, :cos, :tan, :inv, :min, :max, :ifelse,
    :iszero, :isfinite, :isnan, :identity, :size, :axes, :length, :firstindex,
    :lastindex, :eltype, :one, :zero, :<, :>, :<=, :>=, :(==), :(!=), :Bool,
    :Int, :Int8, :Int16, :Int32, :Int64, :UInt, :UInt8, :UInt16, :UInt32,
    :UInt64, :Float16, :Float32, :Float64, :ComplexF32, :ComplexF64))

function readonly_callable(p, ex)
    name = if ex isa Symbol
        ex
    elseif ex isa GlobalRef
        ex.name
    else
        nothing
    end
    isnothing(name) && return false
    startswith(string(name), ".") && (name = Symbol(chop(string(name); head=1, tail=0)))
    name in READONLY || return false
    original = ex isa Symbol && startswith(string(ex), ".") ? name : ex
    f = callee(p, original)
    return !isnothing(f) && isdefined(Base, name) && f === getfield(Base, name)
end

function readonly_expression(p, ex)
    ex isa Expr || return true
    ex.head in (:quote, :inert) && return true
    ex.head === :. && last(ex.args) isa QuoteNode && return readonly_expression(p, first(ex.args))
    ex.head in (:ref, :tuple, :parameters, :kw, :(...)) &&
        return all(a -> readonly_expression(p, a), ex.args)
    c = something(_call(ex), _dotcall(ex), (; f=nothing, args=Any[]))
    return readonly_callable(p, c.f) && all(a -> readonly_expression(p, a), c.args)
end

parents(::Independent, args) = Int[]
parents(::DependsOnAll, args) = unique(reduce(vcat, operands.(args); init=Int[]))
function parents(rule::DependsOn, args)
    return unique(reduce(vcat, (operands(args[i]) for i in rule.positions); init=Int[]))
end
owns(::Independent) = true
owns(::DependsOnAll) = false
owns(rule::DependsOn) = isempty(rule.positions)

# Syntax that must stay embedded (notably indices containing `end`) still uses
# resolved value references. Function/property names and quoted data are not uses.
function inline(p, ex)
    ex isa Symbol && return ex in (:end, :(:), :true, :false, :nothing) ? ex : input!(p, ex)
    ex isa Expr || return ex
    ex.head in (:quote, :inert) && return ex
    if ex.head === :call
        f = callee(p, first(ex.args))
        target = isnothing(f) ? inline(p, first(ex.args)) : first(ex.args)
        return Expr(:call, target, map(a -> inline(p, a), ex.args[2:end])...)
    elseif ex.head === :. && last(ex.args) isa QuoteNode
        return Expr(:., inline(p, first(ex.args)), last(ex.args))
    elseif ex.head === :kw
        return Expr(:kw, first(ex.args), inline(p, last(ex.args)))
    elseif ex.head === :macrocall
        # Macro argument syntax is opaque; preserve it verbatim. Its operands
        # are recorded separately by the parser's opaque operation below.
        return ex
    end
    return Expr(ex.head, map(a -> inline(p, a), ex.args)...)
end

function call!(p, ex)
    f = first(ex.args)
    callable = callee(p, f)
    if callable === Base.maybeview
        args = map(a -> parse!(p, a), ex.args[2:end])
        return operation!(p, IndexOp{false}(value_id(first(args))), Expr(:call, f, args...);
            owned=true,
            readonly=all(a -> readonly_expression(p, a), ex.args[2:end]))
    end
    target = isnothing(callable) ? parse!(p, f) : f
    args = arguments!(a -> argument!(p, a), ex.args[2:end])
    positional = filter(a -> !(a isa Expr && a.head === :parameters), args)
    rule = _lifetime_dependencies(callable, Val(length(positional)))
    unknown = rule isa UnknownDependencies
    splatted = any(a -> a isa Expr && a.head === :(...), positional)
    # Positional rules (including arity-dependent unary arithmetic) cannot be
    # resolved through a dynamic splat. Keep those inputs through scope exit.
    deferred = unknown || (splatted && rule isa DependsOn)
    dependencies = if deferred
        unique(reduce(vcat, operands.(args); init=operands(target)))
    else
        parents(rule, positional)
    end
    return operation!(p, CallOp(), Expr(:call, target, args...);
        owned=unknown || (!deferred && owns(rule)), deferred,
        parents=dependencies, readonly=readonly_callable(p, f))
end

# Julia stores keyword parameters first in the AST, but evaluates positional
# arguments first. Keep that evaluation order while retaining the call layout.
function arguments!(parse_argument, args)
    parsed = Vector{Any}(undef, length(args))
    keywords = findall(a -> a isa Expr && a.head === :parameters, args)
    for i in eachindex(args)
        i in keywords || (parsed[i] = parse_argument(args[i]))
    end
    for i in keywords
        parsed[i] = parse_argument(args[i])
    end
    return parsed
end

function argument!(p, ex)
    ex isa Expr || return parse!(p, ex)
    ex.head === :kw && return Expr(:kw, first(ex.args), parse!(p, last(ex.args)))
    ex.head === :parameters && return Expr(:parameters, map(a -> argument!(p, a), ex.args)...)
    ex.head === :(...) && return Expr(:(...), parse!(p, only(ex.args)))
    return parse!(p, ex)
end

function slice!(p, ex; view=false)
    # Assignments inside an index must retain Julia's native `end` scope and
    # binding order. Treat the whole indexing operation conservatively.
    isempty(parentmodule(@__MODULE__)._assigned_symbols(ex)) ||
        return opaque!(p, view ? :(Base.@view $ex) : ex)
    root = parse!(p, first(ex.args))
    term = Expr(:ref, root, map(a -> inline(p, a), ex.args[2:end])...)
    expression = view ? :(Base.@view $term) : term
    return operation!(p, IndexOp{view}(value_id(root)), expression; owned=true,
        readonly=readonly_expression(p, ex))
end

function lazy!(p, ex, slices::Dict{Any,ValueRef})
    ex isa Expr || return parse!(p, ex)
    if ex.head === :ref
        if !readonly_expression(p, ex)
            empty!(slices)
            return slice!(p, ex)
        end
        # Include resolved values, not just source spelling: a name may have
        # changed between two otherwise identical indexing expressions.
        return get!(slices, inline(p, ex)) do
            return slice!(p, ex)
        end
    elseif _is_broadcast_syntax(ex)
        return broadcast_tree!(p, ex, slices)
    end
    return eager_argument!(p, ex, slices)
end

function eager_argument!(p, ex, slices)
    start = length(p.schedule)
    value = parse!(p, ex)
    any(i -> !p.operations[i].readonly, p.schedule[(start + 1):end]) && empty!(slices)
    return value
end

function broadcast_tree!(p, ex, slices)
    call = something(_call(ex), _dotcall(ex))
    f = ex.head === :call ? Symbol(chop(string(call.f); head=1, tail=0)) : call.f
    # Evaluate a dynamic callable before its arguments, including assignments
    # inside those arguments which could replace the callable's source name.
    dynamic = isnothing(callee(p, f))
    target = dynamic ? eager_argument!(p, f, slices) : f
    args = arguments!(a -> lazy!(p, a, slices), call.args)
    ex.head === :call && !dynamic && return Expr(:call, call.f, args...)
    return Expr(:., target, Expr(:tuple, args...))
end

assign!(p, lhs::Symbol, value) = bind!(p, lhs, value)
assign!(p, lhs::Expr, value) = assign_syntax!(p, Val(lhs.head), lhs, value)
assign_syntax!(p, ::Val{:(...)}, lhs, value) = assign!(p, only(lhs.args), value)

function assign_syntax!(p, ::Val{:(::)}, lhs, value)
    name, type = lhs.args
    p.declared_types[name] = parse!(p, type)
    return bind!(p, name, value)
end

function assign_syntax!(p, ::Val{:tuple}, lhs, value)
    # Use Julia's iteration protocol, including slurping, exactly once.
    names = [gensym(:component) for _ in lhs.args]
    targets = [
        arg isa Expr && arg.head === :(...) ? Expr(:(...), name) : name
        for (arg, name) in zip(lhs.args, names)
    ]
    op = operation!(p, TupleOp(), Expr(:(=), Expr(:tuple, targets...), value);
        parents=operands(value), readonly=false)
    for (target, name) in zip(lhs.args, names)
        assign!(p, target, operation!(p, ProjectionOp(), name; parents=[op.id], readonly=true))
    end
    return value
end

function assign_syntax!(p, ::Val, lhs, value)
    destination = inline(p, lhs)
    roots = operands(destination)
    operation!(p, WriteOp{false}(), Expr(:(=), destination, value);
        parents=operands(value), destination=isempty(roots) ? 0 : first(roots))
    return value
end

parse!(p, ex) = ex
parse!(p, ex::Symbol) = ex in (:true, :false, :nothing, :(:), :end) ? ex : input!(p, ex)
parse!(p, ex::Expr) = parse_syntax!(p, Val(ex.head), ex)
parse_syntax!(p, ::Union{Val{:quote},Val{:inert}}, ex) = ex
parse_syntax!(p, ::Val{:(=)}, ex) = assign!(p, first(ex.args), parse!(p, last(ex.args)))
parse_syntax!(p, ::Val{:ref}, ex) = slice!(p, ex)

function parse_syntax!(p, ::Val{:block}, ex)
    result = nothing
    for stmt in ex.args
        stmt isa LineNumberNode && (p.source=stmt; continue)
        p.statement += 1
        result = parse!(p, stmt)
    end
    return result
end

function parse_syntax!(p, ::Val{:call}, ex)
    _is_broadcast_syntax(ex) && return broadcast!(p, ex)
    if _is_scalar_expression(ex) && readonly_callable(p, first(ex.args))
        return operation!(p, CallOp(), inline(p, ex); owned=true,
            readonly=readonly_expression(p, ex))
    end
    return call!(p, ex)
end

function parse_syntax!(p, ::Val{:macrocall}, ex)
    parentmodule(@__MODULE__)._base_macro(first(ex.args), p.caller, Symbol("@view")) &&
        return slice!(p, last(ex.args); view=true)
    return opaque!(p, ex)
end

function parse_syntax!(p, ::Union{Val{:tuple},Val{:parameters}}, ex)
    term = Expr(ex.head, map(a -> argument!(p, a), ex.args)...)
    return operation!(p, TupleOp(), term; parents=operands(term), readonly=true)
end

function parse_syntax!(p, ::Union{Val{:.},Val{Symbol("'")},Val{:(::)}}, ex)
    _is_broadcast_syntax(ex) && return broadcast!(p, ex)
    return borrow!(p, Expr(ex.head, parse!(p, first(ex.args)), ex.args[2:end]...))
end

borrow!(p, term) = operation!(p, BorrowOp(), term; parents=operands(term), readonly=true)

function broadcast!(p, ex)
    term = lazy!(p, ex, Dict{Any,ValueRef}())
    return operation!(p, BroadcastOp(), term; owned=true, readonly=readonly_expression(p, ex))
end

function broadcast_write!(p, ex)
    lhs, rhs = ex.args
    dest = lhs isa Expr && lhs.head === :ref ? slice!(p, lhs; view=true) : parse!(p, lhs)
    rhs = lazy!(p, rhs, Dict{Any,ValueRef}())
    root = dest isa ValueRef ? dest.id : 0
    if root != 0
        producer = p.operations[p.values[root].producer]
        producer.kind isa IndexOp && (root = producer.kind.root)
    end
    return operation!(p, WriteOp{true}(), Expr(ex.head, dest, rhs);
        parents=operands(dest), destination=root)
end

function parse_syntax!(p, ::Val, ex)
    !isnothing(_broadcast_assignment(ex)) && return broadcast_write!(p, ex)
    update = _compound_assignment(ex)
    if !isnothing(update) && update.lhs isa Symbol
        return assign!(p, update.lhs, call!(p, Expr(:call, update.op, update.lhs, update.rhs)))
    end
    return opaque!(p, ex)
end

function opaque!(p, ex)
    # Opaque macros keep their syntax and effects. A macro that rebinds a name
    # invalidates the old value ID, so subsequent reads observe its new binding.
    ids = [p.bindings[s].id for s in walk_symbols(ex) if haskey(p.bindings, s)]
    ref = operation!(p, OpaqueOp(), OpaqueSyntax(ex, ids); owned=true, deferred=true, parents=ids)
    for name in parentmodule(@__MODULE__)._assigned_symbols(ex)
        delete!(p.bindings, name)
    end
    return ref
end

function parse_scope(scope; caller=parentmodule(@__MODULE__), protected=Set{Symbol}(),
    locals=Set{Symbol}(), policy=LocalScope())
    p = Program(caller, locals, protected, policy)
    return finish_scope!(p, scope)
end

function scope_boundary!(p, header)
    push!(
        p.operations,
        Operation(ScopeOp(), header, operands(header), Int[],
            p.statement, false, 0, true),
    )
    push!(p.schedule, length(p.operations))
    return nothing
end

function parse_let(header, body; caller, protected, locals)
    p = Program(caller, locals, protected, LocalScope())
    bindings = header isa Expr && header.head === :block ? header.args : Any[header]
    for binding in bindings
        binding isa LineNumberNode && continue
        push!(p.locals, binding isa Symbol ? binding : first(binding.args))
    end
    names = Set{Symbol}()
    for binding in bindings
        binding isa LineNumberNode && (p.source=binding; continue)
        p.statement += 1
        if binding isa Symbol
            push!(names, binding)
            delete!(p.bindings, binding)
            scope_boundary!(p, binding)
        else
            name, rhs = binding.args
            value = parse!(p, rhs)
            value isa ValueRef || (value = operation!(p, BorrowOp(), value; readonly=true))
            p.bindings[name] = value
            push!(names, name)
            scope_boundary!(p, Expr(:(=), name, value))
        end
    end
    # Body locals begin after every header initializer has been evaluated.
    bodylocals = setdiff(parentmodule(@__MODULE__)._assigned_symbols(body), names)
    scope_boundary!(p, Expr(:block, sort!(collect(bodylocals); by=string)...))
    foreach(name -> delete!(p.bindings, name), bodylocals)
    return finish_scope!(p, body)
end

function finish_scope!(p, scope)
    p.result = parse!(p, scope isa Expr && scope.head === :block ? scope : Expr(:block, scope))
    statements = _flatten_statements(scope)
    if !isempty(statements)
        terminal = something(
            _assignment(last(statements)), _broadcast_assignment(last(statements)), (; lhs=nothing)
        )
        terminal.lhs isa Expr && terminal.lhs.head === :ref && (p.result = nothing)
    end
    for id in operands(p.result)
        p.values[id].escapes = true
    end
    return rebuild!(p)
end
