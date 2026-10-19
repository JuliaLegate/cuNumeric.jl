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
    function visit(node)
        node isa Expr || return nothing
        node.head in (:quote, :inert) && return nothing
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
        foreach(visit, node.args)
        return nothing
    end
    visit(body)
    return nothing
end

# Function args are caller-owned: protected roots, never freed or fused away.
function _argument_symbols(def)
    names = Set{Symbol}()
    function binding(arg)
        arg isa Symbol && (push!(names, arg); return nothing)
        arg isa Expr || return nothing
        if arg.head in (:tuple, :parameters)
            foreach(binding, arg.args)
        elseif arg.head in (:(::), :(...), :(<:), :(>:))
            binding(first(arg.args))
        end
    end
    for arg in Iterators.flatten((get(def, :args, Any[]), get(def, :kwargs, Any[])))
        name, _, _, _ = MacroTools.splitarg(arg)
        binding(name)
    end
    foreach(binding, get(def, :whereparams, Any[]))
    return names
end

# Symbols a scope reads but never assigns. Like function arguments they are
# stable throughout, so cross-write fusion may guard them with alias checks.
_free_symbols(body) = setdiff(Set{Symbol}(walk_symbols(body)), _assigned_symbols(body))

# Function form: validate, expand `@.`, normalize trailing `return`, run the
# lifetime/fusion passes protecting `protected_roots` (the args).
function _accelerate_rewrite(
    body, caller::Module, protected_roots::Set{Symbol}; aggressive::Bool=false
)
    _reject_nonstraightline(body)
    body = _normalize_return(_expand_dot_macros(body, caller))
    on_rewrite = BCAST_FUSION_DEBUG[] ? AccelerateGraph.log_rewrite : nothing
    return process_ndarray_scope(body; on_rewrite, protected_roots, aggressive, caller)
end

# `begin`/expr form: 1:1 Julia scope (no `let`) — named bindings stay live.
# On GPU, same-shape chains fuse into one multi-output launch; otherwise only
# anonymous temporaries (slices) are freed.
function _accelerate_block_soft(block, caller::Module; aggressive::Bool=false)
    _reject_nonstraightline(block)
    nb = _normalize_return(_expand_dot_macros(block, caller))
    on_rewrite = BCAST_FUSION_DEBUG[] ? AccelerateGraph.log_rewrite : nothing
    return process_ndarray_scope(
        nb; on_rewrite, protected_roots=union(_assigned_symbols(nb), _free_symbols(nb)),
        aggressive, caller, visible=true, local_bindings=_assigned_symbols(nb),
    )
end

# `let` form: hard scope. Full analysis — combine single-use producers, free
# every non-returned temp — re-wrapped in a `let` so only the result escapes.
function _accelerate_block_hard(letexpr, caller::Module; aggressive::Bool=false)
    _reject_nonstraightline(letexpr)
    header, body = length(letexpr.args) == 1 ? (Expr(:block), only(letexpr.args)) : letexpr.args
    header = _expand_dot_macros(header, caller)
    nb = _normalize_return(_expand_dot_macros(body, caller))
    on_rewrite = BCAST_FUSION_DEBUG[] ? AccelerateGraph.log_rewrite : nothing
    allsyntax = Expr(:block, header, nb)
    protected = _free_symbols(allsyntax)
    p = AccelerateGraph.parse_let(header, nb; caller, protected,
        locals=_assigned_symbols(allsyntax))
    plan = AccelerateGraph.optimize(p; fusion=FUSE_BROADCAST_EXPRS, aggressive, on_rewrite)
    return Expr(:let, Expr(:block), AccelerateGraph.emit(plan))
end

# Drop the leading dot: `.+` -> `+`, `.^` -> `^`.
_undot(op::Symbol) = Symbol(chop(string(op); head=1, tail=0))

# AST `@accelerate` emits (pre-`esc`); shared with `@show_lifetimes`. Dispatch:
# function def / `let` (hard scope) / `begin`-expr (soft, 1:1 Julia scope).
function _accelerate_expand(input, caller::Module; aggressive::Bool=false)
    policy = aggressive ? :(cuNumeric.AdaptiveOffsets()) : :(cuNumeric.FixedOffsets{Int64}())
    if MacroTools.isdef(input)
        def = MacroTools.splitdef(input)
        def[:body] = _accelerate_rewrite(
            def[:body], caller, _argument_symbols(def); aggressive
        )
        def[:body] = _lower_broadcast_policy(def[:body], policy)
        return MacroTools.combinedef(def)
    elseif input isa Expr && input.head === :let
        return _lower_broadcast_policy(_accelerate_block_hard(input, caller; aggressive), policy)
    end
    return _lower_broadcast_policy(_accelerate_block_soft(input, caller; aggressive), policy)
end

# Final pass, after fusion and lifetime analysis. Only explicit dotted syntax
# and generated fusion calls acquire the policy; ordinary callees stay unchanged.
function _policy_broadcasted(ex, policy)
    _is_broadcast_syntax(ex) || return _lower_broadcast_policy(ex, policy)
    f, args = if ex.head === :call
        _undot(ex.args[1]), ex.args[2:end]
    else
        _lower_broadcast_policy(ex.args[1], policy), ex.args[2].args
    end
    if f === :^ && length(args) == 2 && args[2] isa Integer
        return :(Base.broadcasted(Base.literal_pow, ^,
            $(_policy_broadcasted(args[1], policy)), Base.Val($(args[2]))))
    end
    lowered = map(a -> _policy_broadcasted(a, policy), args)
    # Keyword arguments belong before the function in a Julia call AST.
    if !isempty(lowered) && lowered[1] isa Expr && lowered[1].head === :parameters
        return Expr(:call, :(Base.broadcasted), lowered[1], f, lowered[2:end]...)
    end
    return Expr(:call, :(Base.broadcasted), f, lowered...)
end

function _lower_broadcast_policy(ex, policy)
    ex isa Expr || return ex
    ex.head in (:quote, :inert) && return ex
    if ex.head === :call && ex.args[1] in (
        :(cuNumeric.copyto_fused_multi!), :(cuNumeric.copyto_fused_multi_alloc!),
        :(cuNumeric.copyto_fused_siblings!),
    )
        return Expr(:call, ex.args[1], policy,
            map(a -> _lower_broadcast_policy(a, policy), ex.args[2:end])...)
    end
    head = string(ex.head)
    if startswith(head, ".") && endswith(head, "=") && length(ex.args) == 2
        lhs, rhs = ex.args
        dest = gensym(:broadcast_dest)
        destexpr = lhs isa Expr && lhs.head === :ref ? :(Base.@view $lhs) : lhs
        destexpr = _lower_broadcast_policy(destexpr, policy)
        bc = if ex.head === :(.=)
            if _is_broadcast_syntax(rhs)
                _policy_broadcasted(rhs, policy)
            else
                :(Base.broadcasted(identity, $(_lower_broadcast_policy(rhs, policy))))
            end
        else
            op = Symbol(chop(head; head=1, tail=1))
            :(Base.broadcasted($op, $dest, $(_policy_broadcasted(rhs, policy))))
        end
        return :(
            let $dest = $destexpr
                cuNumeric._materialize_with_policy!($policy, $dest, $bc)
            end
        )
    end
    if _is_broadcast_syntax(ex)
        return :(cuNumeric._materialize_with_policy($policy, $(_policy_broadcasted(ex, policy))))
    end
    return Expr(ex.head, map(a -> _lower_broadcast_policy(a, policy), ex.args)...)
end

function _accelerate_options(options)
    aggressive = false
    for option in options
        MacroTools.@capture(option, aggressive = value_) && value isa Bool || error(
            "@accelerate: unknown option `$option`; expected `aggressive=true` or `aggressive=false`"
        )
        aggressive = value
    end
    return (; aggressive)
end

@doc"""
    @accelerate function f(args...) ... end
    @accelerate begin ... end
    @accelerate let ... end
    @accelerate expr
    @accelerate aggressive=true function f(args...) ... end

Optimize straight-line array code by coordinating CUDA broadcast fusion within
expressions, fusion across broadcast statements, and scope-aware cleanup of
materialized temporaries. Control flow and nested/anonymous functions are
rejected.

Single-use producers may fuse across intervening read-only calculations when
their inputs are unchanged. Unknown calls remain barriers. Function-form fusion
across writes to other array arguments checks storage overlap at runtime and
retains materialized intermediates when those arguments overlap.

Lifetime analysis uses declared borrowing rules for recognized calls, including
lazy transpose/adjoint wrappers. It does not inspect called function bodies.
Unknown calls are assumed to return locally owned results and not retain their
arguments through side effects: their results and inputs stay live until scope
exit, and are preserved if needed by the returned value. A function that returns
a caller-owned alias cannot be identified automatically. Keep such calls outside
accelerated regions unless the package supplies a borrowing rule. Exceptional
exits rely on normal garbage collection.

Four forms determine which values must remain valid:

  * **function** (preferred): arguments and returned values are protected;
    non-returned locals may fuse into consumers or be freed after their last use.
  * **`begin`**: creates no new Julia scope, so every named binding stays live;
    eligible GPU chains may use one multi-output kernel that materializes them.
  * **`let`**: creates a local scope; only the result escapes, so other locals may
    fuse away or be freed after their last use.
  * **expression**: materializes and returns one expression; eligible operations
    fuse within it and transient temporaries are released.

```julia
@accelerate function step(u, v)   # c may fuse away; the result is returned
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

`aggressive=true` also merges adjacent in-place updates of different arrays,
e.g. `u_new[...] .= f(u, v)` then `v_new[...] .= g(u, v)`, into one launch that
reads shared inputs once. Off by default since the launch maps every input and
output at once, raising peak memory.

It also permits adaptive 32/64-bit element-offset arithmetic in eligible GPU
broadcasts of ranks 1–6 (including single-output and chained broadcasts). Both
variants are compiled per signature; each launch chooses using its local array
dimensions and strides. Default broadcasts use 64-bit offsets. Policies do not
propagate into ordinary function calls, and CPU execution is unchanged.
"""
macro accelerate(args...)
    input = last(args)
    options = _accelerate_options(args[1:(end - 1)])
    return esc(_accelerate_expand(input, __module__; options...))
end

@doc"""
    @show_lifetimes function f(args...) ... end
    @show_lifetimes begin ... end
    @show_lifetimes let ... end

Print the exact expansion [`@accelerate`](@ref) produces for the same input
(all forms), without running it; inserted frees are highlighted. Pure AST work.
"""
macro show_lifetimes(args...)
    input = last(args)
    options = _accelerate_options(args[1:(end - 1)])
    expansion = _accelerate_expand(input, __module__; options...)
    return :(print_lifetime_analysis($(QuoteNode(expansion))))
end
