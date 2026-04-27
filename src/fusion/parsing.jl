# export @fuse

# import ExpressionExplorer: compute_symbols_state
# import MacroTools: postwalk

# struct FusePlan
#     outputs::Tuple{Vararg{Symbol}}
#     inputs::Tuple{Vararg{Symbol}}
#     ops::Tuple{Vararg{Symbol}}
#     scalar_literals::Tuple{Vararg{Number}}
#     lhs::Any
#     rhs::Any
# end

# # -----------------------------------------------------------------------------
# # Runtime argument classification hooks
# # -----------------------------------------------------------------------------

# is_fusion_scalar(::Number) = true
# is_fusion_scalar(_) = false
# is_fusion_array(::NDArray) = true
# is_fusion_array(_) = false
# is_fusion_arg(x) = is_fusion_array(x) || is_fusion_scalar(x)

# function parse_fuse_call(args)
#     isempty(args) && throw(ArgumentError("@fuse expected at least one argument."))

#     call = args[end]
#     kwargs = Dict{Symbol,Any}()
#     for arg in args[1:(end - 1)]
#         if !(arg isa Expr) || !Meta.isexpr(arg, :(=))
#             throw(
#                 ArgumentError(
#                     "Invalid @fuse argument `$(arg)`. Expected keyword form `name=value`.",
#                 ),
#             )
#         end
#         key, value = arg.args
#         if !(key isa Symbol)
#             throw(ArgumentError("Invalid @fuse keyword `$(arg)`; keyword name must be a symbol."))
#         end
#         if key != :blocks && key != :threads
#             throw(
#                 ArgumentError(
#                     "Invalid @fuse keyword `$(key)`. Supported keywords are `blocks` and `threads`.",
#                 ),
#             )
#         end
#         kwargs[key] = value
#     end

#     if !(call isa Expr)
#         throw(
#             ArgumentError(
#                 "@fuse expected a broadcast assignment expression as its final argument; got `$(typeof(call))`.",
#             ),
#         )
#     end

#     return kwargs, call
# end

# function parse_broadcast_assignment(ex::Expr)
#     lhs, rhs = if length(ex.args) == 2
#         ex.args[1], ex.args[2]
#     else
#         throw(
#             ArgumentError(
#                 "@fuse expected a broadcast assignment like `z .= x .+ y .* y`; got malformed expression `$(ex)`.",
#             ),
#         )
#     end

#     lhs_symbol = lhs isa Symbol ? lhs : nothing
#     state = compute_symbols_state(ex)
#     is_plain_assignment = lhs_symbol !== nothing && in(lhs_symbol, state.assignments)

#     if is_plain_assignment || Meta.isexpr(ex, :(=))
#         throw(
#             ArgumentError("@fuse currently supports broadcast assignment `.=` only; got ordinary `=`."),
#         )
#     elseif Meta.isexpr(ex, :(.=))
#         return lhs, rhs
#     end

#     throw(
#         ArgumentError(
#             "@fuse expected a broadcast assignment like `z .= x .+ y .* y`; got expression head `$(ex.head)`.",
#         ),
#     )
# end

# function parse_lhs_outputs(lhs)
#     if lhs isa Symbol
#         return (lhs,), lhs
#     end
#     throw(
#         ArgumentError(
#             "Unsupported @fuse LHS form `$(lhs)`. Supported form for now is a simple symbol target, e.g. `z .= ...`.",
#         ),
#     )
# end

# function collect_rhs_ops(rhs::Expr)
#     ops = Symbol[]
#     # Lifted from fusion_old's collect_ops idea: post-order walk, collect call heads.
#     postwalk(rhs) do node
#         if node isa Expr && (node.head === :call || node.head === :.)
#             op = node.args[1]
#             if op isa Symbol
#                 push!(ops, op)
#             end
#         end
#         return node
#     end
#     return Tuple(ops)
# end

# function collect_rhs_symbol_refs(rhs::Expr)
#     symbol_state = compute_symbols_state(rhs)
#     referenced_symbols = symbol_state.references
#     refs = Symbol[]
#     seen = Set{Symbol}()
#     postwalk(rhs) do node
#         if node isa Symbol &&
#             node !== :true &&
#             node !== :false &&
#             node !== :nothing &&
#             in(node, referenced_symbols) &&
#             !in(node, seen)
#             push!(refs, node)
#             push!(seen, node)
#         end
#         return node
#     end
#     return Tuple(refs)
# end

# function collect_rhs_scalar_literals(rhs::Expr)
#     scalars = Number[]
#     postwalk(rhs) do node
#         if node isa Number
#             push!(scalars, node)
#         elseif node isa QuoteNode && node.value isa Number
#             push!(scalars, node.value)
#         end
#         return node
#     end
#     return Tuple(scalars)
# end

# function build_fuse_plan(ex::Expr)
#     lhs, rhs = parse_broadcast_assignment(ex)
#     outputs, lhs_expr = parse_lhs_outputs(lhs)
#     rhs_expr = rhs isa Expr ? rhs : Expr(:block, rhs)
#     lhs_symbol_state = compute_symbols_state(lhs_expr)
#     rhs_symbol_state = compute_symbols_state(rhs_expr)
#     inputs = collect_rhs_symbol_refs(rhs_expr)
#     ops = collect_rhs_ops(rhs_expr)
#     scalar_literals = collect_rhs_scalar_literals(rhs_expr)

#     shared_names = intersect(lhs_symbol_state.references, rhs_symbol_state.references)
#     if !isempty(shared_names)
#         throw(
#             ArgumentError(
#                 "LHS and RHS of @fuse must not share variables. Shared symbol(s): $(collect(shared_names)).",
#             ),
#         )
#     end

#     return FusePlan(outputs, inputs, ops, scalar_literals, lhs_expr, rhs_expr)
# end

# function validate_fuse_args(plan::FusePlan, outputs::Tuple, inputs::Tuple)
#     if length(outputs) != length(plan.outputs)
#         throw(
#             ArgumentError("Plan expects $(length(plan.outputs)) output(s), got $(length(outputs)).")
#         )
#     end
#     if length(inputs) != length(plan.inputs)
#         throw(ArgumentError("Plan expects $(length(plan.inputs)) input(s), got $(length(inputs))."))
#     end

#     for (name, output) in zip(plan.outputs, outputs)
#         if is_fusion_scalar(output)
#             throw(
#                 ArgumentError(
#                     "Output `$(name)` is scalar-like. Scalar outputs are not supported; use a 0D store/array."
#                 ),
#             )
#         elseif !is_fusion_array(output)
#             throw(ArgumentError("Output `$(name)` has unsupported type `$(typeof(output))`."))
#         end
#     end

#     for (name, input) in zip(plan.inputs, inputs)
#         if !is_fusion_arg(input)
#             throw(
#                 ArgumentError(
#                     "Input `$(name)` has unsupported type `$(typeof(input))`; expected array-like or scalar-like value."
#                 ),
#             )
#         end
#     end

#     return nothing
# end

# """
#     @fuse z .= rhs
#     @fuse blocks=blocks threads=threads z .= rhs

# Parse a broadcast assignment into a `FusePlan` that captures output symbols,
# RHS symbol references, operation ordering, and scalar literals.
# """
# macro fuse(args...)
#     _, call = parse_fuse_call(args)
#     return esc(:(cuNumeric.build_fuse_plan($(QuoteNode(call)))))
# end
