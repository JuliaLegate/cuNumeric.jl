export @fuse, @fuse_plan

"""
    ALLOWED_FUSION_OPS

Set of scalar operations that the fusion frontend is allowed to place in the
fusion IR. Extend this set as the backend learns how to lower more operations.

The parser normalizes dotted operators/functions, so `.+` and `+` both appear
as `:+` in the IR. Whether an operation was dotted is still preserved on each
`FuseCall` node.
"""
const ALLOWED_FUSION_OPS = Set{Symbol}([
    :+, :-, :*, :/, :^,
    :abs, :abs2,
    :sqrt, :cbrt,
    :exp, :exp2, :expm1,
    :log, :log2, :log10, :log1p,
    :sin, :cos, :tan,
    :asin, :acos, :atan,
    :sinh, :cosh, :tanh,
    :min, :max,
])

const FUSE_OPTION_NAMES = Set{Symbol}([:blocks, :threads])

# This cache is intentionally separate from any future function-fusion cache.
# The key is based on the parsed expression IR plus the runtime argument types.
const FUSED_BROADCAST_CACHE = Dict{UInt64,Any}()
const FUSED_BROADCAST_CACHE_LOCK = ReentrantLock()

# -----------------------------------------------------------------------------
# Fusion IR
# -----------------------------------------------------------------------------

abstract type FuseNode end

"""
    FuseInput(name, index)

A symbolic input variable appearing in the RHS expression. `index` is the
runtime position of that input in the input tuple passed to `_fuse_broadcast!`.
Repeated uses of the same symbol point to the same index.
"""
struct FuseInput <: FuseNode
    name::Symbol
    index::Int
end

"""
    FuseLiteral(value)

A literal scalar that appeared directly in the fused expression, for example
`2` in `z .= 2 .* x`.
"""
struct FuseLiteral <: FuseNode
    value::Any
end

"""
    FuseCall(op, args, dotted)

A scalar operation in the fused expression tree.

`op` is normalized, so `:.+` becomes `:+`. `dotted` records whether the user
wrote dotted broadcast syntax for this specific call/operator.
"""
struct FuseCall <: FuseNode
    op::Symbol
    args::Vector{FuseNode}
    dotted::Bool
end

"""
    FusePlan(outputs, inputs, ops, rhs, expr_hash)

A parsed broadcast assignment.

For example, `@fuse z .= x .+ y .* y` becomes a plan with
`outputs == [:z]`, `inputs == [:x, :y]`, and an RHS tree rooted at `:+`.
"""
struct FusePlan
    outputs::Vector{Symbol}
    inputs::Vector{Symbol}
    ops::Vector{Symbol}
    rhs::FuseNode
    expr_hash::UInt64
end

"""
    CompiledFusionKernel

Placeholder object returned by the default `compile_fused_broadcast` method.
Replace or overload `compile_fused_broadcast` and `launch_fused_broadcast!` when
hooking this parser up to the real CUDA/cuNumeric backend.
"""
struct CompiledFusionKernel
    plan::FusePlan
    output_types::Tuple
    input_types::Tuple
end

# -----------------------------------------------------------------------------
# Small display helpers
# -----------------------------------------------------------------------------

function Base.show(io::IO, node::FuseInput)
    print(io, "FuseInput(:", node.name, ", ", node.index, ")")
end

function Base.show(io::IO, node::FuseLiteral)
    print(io, "FuseLiteral(", repr(node.value), ")")
end

function Base.show(io::IO, node::FuseCall)
    dot = node.dotted ? "." : ""
    print(io, "FuseCall(:", dot, node.op, ", ", node.args, ")")
end

function Base.show(io::IO, plan::FusePlan)
    print(io, "FusePlan(outputs = ", plan.outputs,
        ", inputs = ", plan.inputs,
        ", ops = ", plan.ops,
        ", expr_hash = ", plan.expr_hash,
        ", rhs = ", plan.rhs,
        ")")
end

# -----------------------------------------------------------------------------
# Frontend parser
# -----------------------------------------------------------------------------

mutable struct ParseContext
    input_indices::Dict{Symbol,Int}
    inputs::Vector{Symbol}
    ops::Vector{Symbol}
end

ParseContext() = ParseContext(Dict{Symbol,Int}(), Symbol[], Symbol[])

function intern_input!(ctx::ParseContext, name::Symbol)
    idx = get(ctx.input_indices, name, 0)
    if idx == 0
        push!(ctx.inputs, name)
        idx = length(ctx.inputs)
        ctx.input_indices[name] = idx
    end
    return idx
end

"""
    build_fuse_plan(ex::Expr) -> FusePlan

Parse a broadcast assignment expression into a reusable fusion plan.

Supported MVP syntax:

```julia
z .= x .+ y .* y
z .= sin.(x) .+ 2 .* y
```

This function intentionally does not compile or launch anything.
"""
function build_fuse_plan(ex::Expr)
    if Meta.isexpr(ex, :(.=))
        lhs, rhs = ex.args
    elseif Meta.isexpr(ex, :(=))
        throw(
            ArgumentError(
                "@fuse currently supports broadcast assignment `.=` only; got ordinary `=`."
            ),
        )
    else
        throw(
            ArgumentError(
                "@fuse expected a broadcast assignment like `z .= x .+ y .* y`; got expression head `$(ex.head)`."
            ),
        )
    end

    outputs = parse_lhs(lhs)
    ctx = ParseContext()
    rhs_node = parse_rhs(rhs, ctx)

    # Keep the full operator occurrence order in the plan. Use `unique_ops(plan)`
    # if the backend only wants each operator once.
    plan_without_hash = FusePlan(outputs, copy(ctx.inputs), copy(ctx.ops), rhs_node, UInt64(0))
    return FusePlan(
        outputs, copy(ctx.inputs), copy(ctx.ops), rhs_node, hash(plan_signature(plan_without_hash))
    )
end

function build_fuse_plan(ex)
    throw(ArgumentError("@fuse expected an expression; got `$(repr(ex))`."))
end

function parse_lhs(lhs)
    if lhs isa Symbol
        return Symbol[lhs]
    elseif Meta.isexpr(lhs, :tuple)
        throw(
            ArgumentError(
                "Multiple-output broadcast fusion is not implemented yet. Use a single output, e.g. `z .= rhs`."
            ),
        )
    elseif Meta.isexpr(lhs, :ref)
        throw(
            ArgumentError(
                "Indexed/view outputs are not implemented yet. Use a whole-array output, e.g. `z .= rhs`."
            ),
        )
    else
        throw(ArgumentError("Unsupported @fuse output expression: `$(lhs)`."))
    end
end

function parse_rhs(ex, ctx::ParseContext)::FuseNode
    if ex isa Symbol
        idx = intern_input!(ctx, ex)
        return FuseInput(ex, idx)
    elseif ex isa Number
        return FuseLiteral(ex)
    elseif ex isa Bool
        return FuseLiteral(ex)
    elseif ex isa Char
        return FuseLiteral(ex)
    elseif ex isa String
        throw(ArgumentError("String literals are not valid scalar fusion literals: `$(ex)`."))
    elseif Meta.isexpr(ex, :call)
        return parse_call(ex, ctx)
    elseif Meta.isexpr(ex, :.)
        return parse_dotted_call(ex, ctx)
    elseif Meta.isexpr(ex, :ref)
        throw(ArgumentError("Indexing inside fused RHS is not implemented yet: `$(ex)`."))
    elseif Meta.isexpr(ex, :tuple)
        throw(ArgumentError("Tuple construction inside fused RHS is not implemented yet: `$(ex)`."))
    elseif Meta.isexpr(ex, :block)
        throw(
            ArgumentError(
                "Block expressions are not supported in broadcast-fusion RHS expressions."
            ),
        )
    else
        throw(
            ArgumentError("Unsupported expression in fused RHS: `$(ex)` with type `$(typeof(ex))`.")
        )
    end
end

function parse_call(ex::Expr, ctx::ParseContext)::FuseNode
    raw_op = ex.args[1]
    op, dotted = normalize_call_operator(raw_op)
    check_allowed_op!(op, raw_op)

    push!(ctx.ops, op)
    args = FuseNode[parse_rhs(arg, ctx) for arg in ex.args[2:end]]
    return FuseCall(op, args, dotted)
end

function parse_dotted_call(ex::Expr, ctx::ParseContext)::FuseNode
    # Dotted function calls like `sin.(x)` usually parse as
    # Expr(:., :sin, Expr(:tuple, :x)). Property access also uses head `:.`, so
    # require the second argument to be an argument tuple.
    if length(ex.args) != 2 || !(ex.args[2] isa Expr) || ex.args[2].head != :tuple
        throw(ArgumentError("Property access is not supported in fused RHS expressions: `$(ex)`."))
    end

    op = function_name_symbol(ex.args[1])
    check_allowed_op!(op, ex.args[1])

    push!(ctx.ops, op)
    args = FuseNode[parse_rhs(arg, ctx) for arg in ex.args[2].args]
    return FuseCall(op, args, true)
end

function normalize_call_operator(raw_op)
    if raw_op isa Symbol
        op_string = String(raw_op)
        if startswith(op_string, ".")
            return Symbol(op_string[2:end]), true
        else
            return raw_op, false
        end
    end

    # Qualified calls such as Base.sin(x). Dotted qualified calls such as
    # Base.sin.(x) are handled by `parse_dotted_call`.
    return function_name_symbol(raw_op), false
end

function function_name_symbol(ex)
    if ex isa Symbol
        return ex
    elseif ex isa QuoteNode && ex.value isa Symbol
        return ex.value
    elseif Meta.isexpr(ex, :.)
        # Handles qualified function names. We intentionally drop the module
        # qualification here because the backend's allow-list is scalar-op based.
        return function_name_symbol(ex.args[end])
    else
        throw(ArgumentError("Could not determine function/operator name from `$(ex)`."))
    end
end

function check_allowed_op!(op::Symbol, raw_op)
    if !(op in ALLOWED_FUSION_OPS)
        throw(
            ArgumentError(
                "Operation `$(raw_op)` normalized to `$(op)`, which is not in ALLOWED_FUSION_OPS."
            ),
        )
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Plan signatures and cache keys
# -----------------------------------------------------------------------------

node_signature(node::FuseInput) = (:input, node.name, node.index)
node_signature(node::FuseLiteral) = (:literal, typeof(node.value), node.value)
node_signature(node::FuseCall) = (:call, node.op, node.dotted, Tuple(node_signature.(node.args)))

function plan_signature(plan::FusePlan)
    return (:broadcast_assignment,
        Tuple(plan.outputs),
        Tuple(plan.inputs),
        Tuple(plan.ops),
        node_signature(plan.rhs))
end

"""
    unique_ops(plan::FusePlan)

Return the operations appearing in the RHS in first-occurrence order.
"""
function unique_ops(plan::FusePlan)
    seen = Set{Symbol}()
    out = Symbol[]
    for op in plan.ops
        if !(op in seen)
            push!(seen, op)
            push!(out, op)
        end
    end
    return out
end

"""
    fusion_cache_key(plan, outputs, inputs) -> UInt64

Cache key for the compiled kernel. Launch options such as `blocks` and
`threads` are intentionally excluded because they should not affect codegen.
"""
function fusion_cache_key(plan::FusePlan, outputs::Tuple, inputs::Tuple)
    return hash((plan.expr_hash, map(typeof, outputs), map(typeof, inputs)))
end

# -----------------------------------------------------------------------------
# Runtime argument classification hooks
# -----------------------------------------------------------------------------

is_fusion_scalar(::Number) = true
is_fusion_scalar(_) = false
is_fusion_array(::AbstractArray) = true
is_fusion_array(_) = false
is_fusion_arg(x) = is_fusion_array(x) || is_fusion_scalar(x)

function validate_fuse_args(plan::FusePlan, outputs::Tuple, inputs::Tuple)
    if length(outputs) != length(plan.outputs)
        throw(
            ArgumentError("Plan expects $(length(plan.outputs)) output(s), got $(length(outputs)).")
        )
    end
    if length(inputs) != length(plan.inputs)
        throw(ArgumentError("Plan expects $(length(plan.inputs)) input(s), got $(length(inputs))."))
    end

    for (name, output) in zip(plan.outputs, outputs)
        if is_fusion_scalar(output)
            throw(
                ArgumentError(
                    "Output `$(name)` is scalar-like. Scalar outputs are not supported; use a 0D store/array."
                ),
            )
        elseif !is_fusion_array(output)
            throw(ArgumentError("Output `$(name)` has unsupported type `$(typeof(output))`."))
        end
    end

    for (name, input) in zip(plan.inputs, inputs)
        if !is_fusion_arg(input)
            throw(
                ArgumentError(
                    "Input `$(name)` has unsupported type `$(typeof(input))`; expected array-like or scalar-like value."
                ),
            )
        end
    end

    return nothing
end

# -----------------------------------------------------------------------------
# Runtime compilation / launch path
# -----------------------------------------------------------------------------

"""
    _fuse_broadcast!(plan, outputs, inputs; blocks=nothing, threads=nothing)

Runtime entrypoint emitted by `@fuse`. This validates runtime arguments,
computes a cache key, compiles if necessary, and launches the backend.
"""
function _fuse_broadcast!(
    plan::FusePlan,
    outputs::Tuple,
    inputs::Tuple;
    blocks=nothing,
    threads=nothing,
)
    validate_fuse_args(plan, outputs, inputs)
    key = fusion_cache_key(plan, outputs, inputs)

    local kernel
    lock(FUSED_BROADCAST_CACHE_LOCK)
    try
        kernel = get!(FUSED_BROADCAST_CACHE, key) do
            compile_fused_broadcast(plan, outputs, inputs)
        end
    finally
        unlock(FUSED_BROADCAST_CACHE_LOCK)
    end

    return launch_fused_broadcast!(kernel, outputs, inputs; blocks=blocks, threads=threads)
end

"""
    compile_fused_broadcast(plan, outputs, inputs)

Backend hook. The default stores type information only. Replace this with the
real CUDATask/PTX generation path.
"""
function compile_fused_broadcast(plan::FusePlan, outputs::Tuple, inputs::Tuple)
    return CompiledFusionKernel(plan, map(typeof, outputs), map(typeof, inputs))
end

"""
    launch_fused_broadcast!(kernel, outputs, inputs; blocks, threads)

Backend hook. The default throws because this file only implements parsing,
validation, and caching.
"""
function launch_fused_broadcast!(
    kernel::CompiledFusionKernel,
    outputs::Tuple,
    inputs::Tuple;
    blocks=nothing,
    threads=nothing,
)
    throw(
        ErrorException(
            "No fusion backend has been installed. The expression parsed and cached successfully, " *
            "but `launch_fused_broadcast!` must be overloaded to call the CUDA/cuNumeric backend.",
        ),
    )
end

# -----------------------------------------------------------------------------
# Macros
# -----------------------------------------------------------------------------

function parse_fuse_macro_args(args)
    isempty(args) && throw(ArgumentError("@fuse expected `z .= rhs`."))

    call = args[end]
    option_exprs = Expr[]

    for opt in args[1:(end - 1)]
        if opt isa Symbol && opt in FUSE_OPTION_NAMES
            # Supports the old style `@fuse blocks threads z .= rhs`, meaning
            # `blocks = blocks, threads = threads`.
            push!(option_exprs, Expr(:kw, opt, opt))
        elseif Meta.isexpr(opt, :(=)) || Meta.isexpr(opt, :kw)
            name = opt.args[1]
            value = opt.args[2]
            if !(name isa Symbol) || !(name in FUSE_OPTION_NAMES)
                throw(
                    ArgumentError(
                        "Invalid @fuse option `$(name)`. Expected one of $(collect(FUSE_OPTION_NAMES))."
                    ),
                )
            end
            push!(option_exprs, Expr(:kw, name, value))
        else
            throw(
                ArgumentError(
                    "Invalid @fuse option/expression `$(opt)`. Expected options followed by `z .= rhs`."
                ),
            )
        end
    end

    return option_exprs, call
end

"""
    @fuse z .= rhs
    @fuse blocks=blocks threads=threads z .= rhs

Parse a broadcast assignment into a `FusePlan`, validate the runtime arguments,
compile/cache the corresponding kernel, and dispatch to `launch_fused_broadcast!`.
"""
macro fuse(args...)
    option_exprs, call = parse_fuse_macro_args(args)
    plan = build_fuse_plan(call)

    output_tuple = Expr(:tuple, plan.outputs...)
    input_tuple = Expr(:tuple, plan.inputs...)
    runtime = GlobalRef(@__MODULE__, :_fuse_broadcast!)

    if isempty(option_exprs)
        return esc(Expr(:call, runtime, QuoteNode(plan), output_tuple, input_tuple))
    else
        return esc(
            Expr(
                :call,
                runtime,
                Expr(:parameters, option_exprs...),
                QuoteNode(plan),
                output_tuple,
                input_tuple,
            ),
        )
    end
end

"""
    @fuse_plan z .= rhs

Parse only. Useful while developing the frontend.
"""
macro fuse_plan(call)
    plan = build_fuse_plan(call)
    return QuoteNode(plan)
end

# For @fuse on user defined function
# function maybe_fuse_kernel(fn::Function, output_indices, args::T; kwargs...) where {T}
#     cache_key = hash((fn, T)) #! ADD KWRAGS TO THIS?

#     if !haskey(FUSED_KERNEL_CACHE, cache_key)
#         FUSED_KERNEL_CACHE[cache_key] = fuse_function(
#             fn,
#             output_indices,
#             args...;
#             kwargs...,
#         )
#     end

#     return cache_key
# end

# function run_fused_kernel(cache_key; blocks, threads)
#     fkd = FUSED_KERNEL_CACHE[cache_key]
#     cuNumeric.launch(fkd.ct, fkd.inputs, fkd.outputs, fkd.scalars; blocks=blocks, threads=threads)
# end

# macro fuse(ex...)
#     call = ex[end]
#     kwargs = map(ex[1:(end - 1)]) do kwarg
#         if kwarg in FUSE_KWARGS
#             :($kwarg = $kwarg)
#         else
#             throw(ArgumentError("Invalid keyword argument '$kwarg', expected one of $(FUSE_KWARGS)"))
#         end
#     end

#     #! TODO SET DEFAULT BLOCKS/THREADS
#     #! HOW TO KNOW WHATS THE RIGHT DIMENSION??

#     blocks = get(kwargs[:blocks], DEFAULT_BLOCKS)
#     threads = get(kwargs[:threads], DEFAULT_THREADS)
#     output_indices = get(kwargs[:output_indices], ())

#     code = quote end

#     if Meta.isexpr(call, :function)
#         # Parses function epxression into args, name etc.
#         data = splitdef(longdef(call))
#         # splitarg parses args into (name, type, is_slurp, default)
#         arg_data = map(splitarg, data[:args])
#         kwarg_data = map(splitarg, data[:kwargs])

#         arg_names = getindex.(arg_data, 1)

#         @gensym wrapper_name, cache_key

#         push!(code.args,
#             quote

#                 # The fused function
#                 function $(wrapper_name)() where {$(dict[:whereparams]...)}
#                     #! NOT SURE THIS IS RIGHT WAY TO INTERPOLATE ARGS
#                     #! WANT IT INTERPOLATE AS TUPLE OF ARGS
#                     #! TODO PASS KWARGS
#                     $cache_key = maybe_fuse_kernel($(data[:name]), $output_indices, $(arg_names...))
#                     run_fused_kernel($cache_key; blocks=($blocks), threads=($threads))
#                 end

#                 # Replace original function with call to fused
#                 #! NOT SURE OVERWRITING LIKE THIS IS GREAT
#                 function $(data[:name])(
#                     $(data[:args]...); $(data[:kwargs]...)
#                 ) where {$(dict[:whereparams]...)}
#                     $(wrapper_name)()
#                 end
#             end
#         )
#     else
#         throw(ArgumentError("fuse expected assignment expression or function, got $(call.head)"))
#     end

#     return esc(quote
#         let
#             $code
#         end
#     end)
# end
