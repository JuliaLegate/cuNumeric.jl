using MacroTools: MacroTools

# Rejected everywhere: control flow makes last-use freeing unsound (a temp freed
# after its textual last use could be revived on another path).
const _CONTROL_FLOW_HEADS = (:if, :elseif, :for, :while, :try, :do, :break, :continue, :&&, :||)

_is_function_lhs(::Any) = false
function _is_function_lhs(lhs::Expr)
    lhs.head === :call && return true
    lhs.head === :where && return _is_function_lhs(first(lhs.args))
    return false
end

function _reject_nonstraightline(body)
    MacroTools.postwalk(body) do node
        node isa Expr || return node
        if node.head === :function || node.head === :-> ||
            (node.head === :(=) && _is_function_lhs(first(node.args)))
            error("@accelerate: nested/anonymous function definitions are not supported")
        end
        if node.head in _CONTROL_FLOW_HEADS
            error(
                "@accelerate: control flow (`$(node.head)`) is not supported; " *
                "only straight-line code can be accelerated",
            )
        end
        return node
    end
    return nothing
end

# Function args are caller-owned: protected roots, never freed or fused away.
function _argument_symbols(def)
    names = Set{Symbol}()
    for arg in Iterators.flatten((get(def, :args, Any[]), get(def, :kwargs, Any[])))
        name, _, _, _ = MacroTools.splitarg(arg)
        name isa Symbol && push!(names, name)
    end
    return names
end

# Function form: validate, expand `@.`, normalize trailing `return`, run the
# lifetime/fusion passes protecting `protected_roots` (the args).
function _accelerate_rewrite(body, caller::Module, protected_roots::Set{Symbol})
    _reject_nonstraightline(body)
    body = _normalize_return(_expand_dot_macros(body, caller))
    on_rewrite = BCAST_FUSION_DEBUG[] ? InterBroadcastFusion.log_rewrite : nothing
    return process_ndarray_scope(body; on_rewrite, protected_roots)
end

# `begin`/expr form: 1:1 Julia scope (no `let`) — named bindings stay live.
# On GPU, same-shape chains fuse into one multi-output launch; otherwise only
# anonymous temporaries (slices) are freed.
function _accelerate_block_soft(block, caller::Module)
    _reject_nonstraightline(block)
    nb = _normalize_return(_expand_dot_macros(block, caller))
    on_rewrite = BCAST_FUSION_DEBUG[] ? InterBroadcastFusion.log_rewrite : nothing
    fallback = process_ndarray_scope(
        nb; on_rewrite, protected_roots=_assigned_symbols(nb)
    )
    @static if FUSE_BROADCAST_EXPRS && HAS_CUDA
        fused = _try_fuse_block_multi(nb)
        if !isnothing(fused)
            return quote
                if cuNumeric._has_gpu_target()
                    $fused
                else
                    $fallback
                end
            end
        end
    end
    # Protect named bindings; free only anonymous temps.
    return fallback
end

# Flatten a `let` node's bindings + body into one statement block.
function _let_body(letexpr::Expr)
    stmts = Any[]
    for part in letexpr.args
        if part isa Expr && part.head === :block
            append!(stmts, part.args)
        elseif part isa Expr && part.head === :(=)
            push!(stmts, part)
        elseif !isnothing(part)
            push!(stmts, part)
        end
    end
    return Expr(:block, stmts...)
end

# `let` form: hard scope. Full analysis — combine single-use producers, free
# every non-returned temp — re-wrapped in a `let` so only the result escapes.
function _accelerate_block_hard(letexpr, caller::Module)
    body = _let_body(letexpr)
    _reject_nonstraightline(body)
    nb = _normalize_return(_expand_dot_macros(body, caller))
    on_rewrite = BCAST_FUSION_DEBUG[] ? InterBroadcastFusion.log_rewrite : nothing
    rewritten = process_ndarray_scope(nb; on_rewrite, protected_roots=Set{Symbol}())
    bindings = union(_assigned_symbols(nb), _assigned_symbols(rewritten))
    return _lexical_scope(rewritten, bindings)
end

# Drop the leading dot: `.+` -> `+`, `.^` -> `^`.
_undot(op::Symbol) = Symbol(chop(string(op); head=1, tail=0))

# Dotted RHS -> lazy `Base.broadcasted(...)`: chain vars become `MatRef{k}`,
# slice leaves are hoisted into `hoisted` (temp => slice) to free post-launch.
# `nothing` when not lowerable (caller falls back).
function _to_broadcasted(expr, idx::AbstractDict{Symbol,Int}, hoisted::Vector)
    if expr isa Symbol
        haskey(idx, expr) && return :(cuNumeric.MatRef($(idx[expr])))
        return expr
    end
    expr isa Expr || return expr
    if expr.head === :ref
        # Slice temp: bail if it indexes a chain var, else hoist to free later.
        any(s -> haskey(idx, s), walk_symbols(expr)) && return nothing
        tmp = gensym(:slice)
        push!(hoisted, tmp => expr)
        return tmp
    end
    if expr.head === :call && expr.args[1] isa Symbol && _is_broadcast_op(expr.args[1])
        cargs = map(a -> _to_broadcasted(a, idx, hoisted), expr.args[2:end])
        any(isnothing, cargs) && return nothing
        return Expr(:call, :(Base.broadcasted), _undot(expr.args[1]), cargs...)
    end
    if expr.head === :. && length(expr.args) == 2 &&
        expr.args[2] isa Expr && expr.args[2].head === :tuple
        cargs = map(a -> _to_broadcasted(a, idx, hoisted), expr.args[2].args)
        any(isnothing, cargs) && return nothing
        return Expr(:call, :(Base.broadcasted), expr.args[1], cargs...)
    end
    # Non-dotted scalar leaf; unsafe if it reads a chain var as a scalar.
    any(s -> haskey(idx, s), walk_symbols(expr)) && return nothing
    return expr
end

function _is_top_broadcast(rhs)
    return (
        rhs isa Expr && rhs.head === :call && rhs.args[1] isa Symbol &&
        _is_broadcast_op(rhs.args[1])
    ) ||
           (
        rhs isa Expr && rhs.head === :. && length(rhs.args) == 2 &&
        rhs.args[2] isa Expr && rhs.args[2].head === :tuple
    )
end

# SSA chain of `sym = <broadcast>` (+ optional trailing return) -> one
# multi-output launch materializing each result. `nothing` -> caller falls back.
function _try_fuse_block_multi(block)
    stmts = _scope_statements(block)
    isnothing(stmts) && return nothing
    stmts = filter(s -> !(s isa LineNumberNode), stmts)
    isempty(stmts) && return nothing

    assigns = stmts
    ret = nothing
    if isnothing(_assignment(last(stmts)))
        ret = last(stmts)
        assigns = stmts[1:(end - 1)]
    end
    length(assigns) >= 2 || return nothing

    syms = Symbol[]
    idx = Dict{Symbol,Int}()
    seg_exprs = Any[]
    hoisted = Pair{Symbol,Any}[]                   # slice temp => slice expr
    for stmt in assigns
        a = _assignment(stmt)
        isnothing(a) && return nothing
        a.lhs isa Symbol || return nothing        # no indexed-assign in this path
        a.lhs in syms && return nothing            # SSA: no reassignment
        _is_top_broadcast(a.rhs) || return nothing # must be a real broadcast
        seg = _to_broadcasted(a.rhs, idx, hoisted)
        isnothing(seg) && return nothing
        push!(seg_exprs, seg)
        push!(syms, a.lhs)
        idx[a.lhs] = length(syms)
    end
    outs = gensym(:outs)
    slice_binds = [:($t = $e) for (t, e) in hoisted]
    slice_frees = [:(cuNumeric.maybe_insert_delete($t)) for (t, _) in hoisted]
    binds = [:($(syms[i]) = $outs[$i]) for i in eachindex(syms)]
    value = isnothing(ret) ? last(syms) : ret
    return quote
        $(slice_binds...)                          # materialize slice views
        $outs = cuNumeric.copyto_fused_multi_alloc!(($(seg_exprs...),))
        $(slice_frees...)                          # free them after the launch
        $(binds...)
        $value
    end
end

# AST `@accelerate` emits (pre-`esc`); shared with `@show_lifetimes`. Dispatch:
# function def / `let` (hard scope) / `begin`-expr (soft, 1:1 Julia scope).
function _accelerate_expand(input, caller::Module)
    if MacroTools.isdef(input)
        def = MacroTools.splitdef(input)
        def[:body] = _accelerate_rewrite(def[:body], caller, _argument_symbols(def))
        return MacroTools.combinedef(def)
    elseif input isa Expr && input.head === :let
        return _accelerate_block_hard(input, caller)
    end
    return _accelerate_block_soft(input, caller)
end

@doc"""
    @accelerate function f(args...) ... end
    @accelerate begin ... end
    @accelerate let ... end
    @accelerate expr

Fuse straight-line array code into fewer kernel launches and free temporaries.
The body must be straight-line — control flow and nested/anonymous functions are
rejected. Four forms, by scope:

  * **function** (preferred): args are caller-owned; only the returned value stays
    materialized, so non-returned intermediates fuse away and are freed.
  * **`begin`**: 1:1 Julia scope — every named binding stays live; on GPU,
    same-shape chains may fuse into one multi-output launch; anonymous temps
    (slices) are freed.
  * **`let`**: hard scope — combines single-use producers and frees every
    non-returned temporary; only the returned value(s) escape. Maximum reuse.
  * **expression**: materializes and returns one expression without introducing
    a new scope.

```julia
@accelerate function step(u, v)   # c freed; w returned
    c = u .* v
    return c .^ 2
end
a, b = @accelerate begin          # a and b both stay live, one GPU launch
    a = x .* y
    b = a .+ 1
    (a, b)
end
result = @accelerate (x .+ y .* z)
```
"""
macro accelerate(input)
    return esc(_accelerate_expand(input, __module__))
end

@doc"""
    @show_lifetimes function f(args...) ... end
    @show_lifetimes begin ... end
    @show_lifetimes let ... end

Print the exact expansion [`@accelerate`](@ref) produces for the same input
(all forms), without running it; inserted frees are highlighted. Pure AST work.
"""
macro show_lifetimes(input)
    expansion = _accelerate_expand(input, __module__)
    return :(print_lifetime_analysis($(QuoteNode(expansion))))
end
