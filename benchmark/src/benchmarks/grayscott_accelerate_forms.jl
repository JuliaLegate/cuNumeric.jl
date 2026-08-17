# Compare the four scope contracts of `@accelerate` on one shared Gray-Scott step.
# Each type has a distinct result name so benchmark runs produce separate CSVs.

abstract type AbstractGrayScottAccelerateForm{T} <: AbstractGrayScott{T} end

Base.@kwdef struct GrayScottFunctionAccelerated{T} <:
                   AbstractGrayScottAccelerateForm{T}
    N::Int
    M::Int
end

Base.@kwdef struct GrayScottBeginAccelerated{T} <: AbstractGrayScottAccelerateForm{T}
    N::Int
    M::Int
end

Base.@kwdef struct GrayScottLetAccelerated{T} <: AbstractGrayScottAccelerateForm{T}
    N::Int
    M::Int
end

Base.@kwdef struct GrayScottExpressionAccelerated{T} <:
                   AbstractGrayScottAccelerateForm{T}
    N::Int
    M::Int
end

name(::GrayScottFunctionAccelerated) = "grayscott_function_accelerated"
name(::GrayScottBeginAccelerated) = "grayscott_begin_accelerated"
name(::GrayScottLetAccelerated) = "grayscott_let_accelerated"
name(::GrayScottExpressionAccelerated) = "grayscott_expression_accelerated"

# Function form is the reusable default: arguments and the return value survive,
# while non-returned locals may fuse across statements or die after their last use.
let body = deepcopy(GRAYSCOTT_STEP_BODY)
    @eval @accelerate function _gs_step!(
        b::GrayScottFunctionAccelerated, u, v, u_new, v_new, args::GSParams
    )
        $body
    end
end

# `begin` adds no scope. Every named local remains visible, so it measures the
# multi-output/materialized path rather than eliminating named intermediates.
let body = deepcopy(GRAYSCOTT_STEP_BODY)
    @eval function _gs_step!(
        b::GrayScottBeginAccelerated, u, v, u_new, v_new, args::GSParams
    )
        @accelerate begin
            $body
        end
    end
end

# `let` is a hard one-off scope. Only its result escapes, allowing aggressive
# inter-statement fusion and last-use cleanup for all other local temporaries.
let body = deepcopy(GRAYSCOTT_STEP_BODY)
    @eval function _gs_step!(
        b::GrayScottLetAccelerated, u, v, u_new, v_new, args::GSParams
    )
        @accelerate let
            $body
        end
    end
end

# Expression form has no multi-statement scope. Accelerating each RHS preserves
# fusion inside that expression but deliberately materializes statement results,
# isolating intra-expression fusion from the inter-statement rewrite cases above.
function accelerate_grayscott_rhs(body::Expr)
    statements = Any[]
    for statement in body.args
        if statement isa LineNumberNode
            push!(statements, statement)
        elseif statement isa Expr && statement.head === :(=)
            lhs, rhs = statement.args
            push!(statements, :($lhs = @accelerate $rhs))
        else
            error("Gray-Scott expression benchmark expected assignments; got $(repr(statement))")
        end
    end
    return Expr(:block, statements...)
end

let body = accelerate_grayscott_rhs(deepcopy(GRAYSCOTT_STEP_BODY))
    @eval function _gs_step!(
        b::GrayScottExpressionAccelerated, u, v, u_new, v_new, args::GSParams
    )
        $body
    end
end

register_benchmark("grayscott_function_accelerated", GrayScottFunctionAccelerated)
register_benchmark("grayscott_begin_accelerated", GrayScottBeginAccelerated)
register_benchmark("grayscott_let_accelerated", GrayScottLetAccelerated)
register_benchmark("grayscott_expression_accelerated", GrayScottExpressionAccelerated)
