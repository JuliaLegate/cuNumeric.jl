module AccelerateGraph

using Graphs
using ..ScopingUtils
using ..cuNumeric: Independent, DependsOn, DependsOnAll, UnknownDependencies,
    _lifetime_dependencies

# Value references are leaves in operation templates. They are never emitted
# into Julia code: emission replaces each reference with its local binding.
struct ValueRef
    id::Int
end

struct OpaqueSyntax
    expression::Any
    inputs::Vector{Int}
end

abstract type OperationKind end
struct CallOp <: OperationKind end
struct InputOp <: OperationKind end
struct BroadcastOp <: OperationKind end
struct IndexOp{View} <: OperationKind end
struct WriteOp{Broadcast} <: OperationKind end
struct BindOp <: OperationKind end
struct ScopeOp <: OperationKind end
abstract type BorrowingOp <: OperationKind end
struct BorrowOp <: BorrowingOp end
struct TupleOp <: BorrowingOp end
struct ProjectionOp <: BorrowingOp end
struct OpaqueOp <: OperationKind end
struct FusedOp <: OperationKind end

mutable struct Operation{K<:OperationKind}
    kind::K
    expression::Any
    inputs::Vector{Int}       # ordered, including repeated operands
    outputs::Vector{Int}
    statement::Int
    readonly::Bool
    destination::Int         # storage root of a write, or zero
    active::Bool
end

mutable struct Value
    name::Symbol
    producer::Int            # stable operation ID, including caller snapshots
    owned::Bool
    deferred::Bool
    escapes::Bool
    materialized::Bool
    external::Union{Nothing,Symbol}
end

abstract type ScopePolicy end
struct LocalScope <: ScopePolicy end
struct VisibleScope <: ScopePolicy end

mutable struct Program{S<:ScopePolicy}
    operations::Vector{Operation}
    values::Vector{Value}
    schedule::Vector{Int}
    dependencies::SimpleDiGraph{Int}
    lifetimes::SimpleDiGraph{Int}
    bindings::Dict{Symbol,ValueRef}
    declared_types::Dict{Symbol,Any}
    locations::Dict{Int,LineNumberNode}
    caller::Module
    locals::Set{Symbol}
    protected::Set{Symbol}
    policy::S
    result::Any
    statement::Int
    source::Any
end

function Program(caller, locals, protected, policy)
    return Program(Operation[], Value[], Int[], SimpleDiGraph(0), SimpleDiGraph(0),
        Dict{Symbol,ValueRef}(), Dict{Symbol,Any}(), Dict{Int,LineNumberNode}(),
        caller, locals, protected, policy, nothing, 0, nothing)
end

function clone(p::Program)
    return Program(deepcopy(p.operations), deepcopy(p.values), copy(p.schedule),
        copy(p.dependencies), copy(p.lifetimes), copy(p.bindings), copy(p.declared_types),
        copy(p.locations), p.caller,
        copy(p.locals), copy(p.protected), p.policy, deepcopy(p.result), p.statement, p.source)
end

operands(x::ValueRef) = [x.id]
operands(x::OpaqueSyntax) = x.inputs
operands(x) = Int[]
operands(x::Expr) = reduce(vcat, operands.(x.args); init=Int[])

replace_values(f, x::ValueRef) = f(x)
replace_values(f, x) = x
replace_values(f, x::Expr) = Expr(x.head, map(a -> replace_values(f, a), x.args)...)

function input!(p, name::Symbol)
    get!(p.bindings, name) do
        ref = operation!(p, InputOp(), name; readonly=true)
        p.values[ref.id].external = name
        p.values[ref.id].escapes = true
        return ref
    end
end

function operation!(p, kind, expression; owned=false, deferred=false,
    parents=Int[], readonly=false, destination=0)
    id = length(p.operations) + 1
    p.source isa LineNumberNode && (p.locations[p.statement] = p.source)
    value = length(p.values) + 1
    push!(p.values, Value(gensym(:value), id, owned, deferred, false, true, nothing))
    add_vertex!(p.lifetimes)
    for parent in unique(parents)
        add_edge!(p.lifetimes, parent, value)
    end
    push!(
        p.operations,
        Operation(kind, expression, operands(expression), [value],
            p.statement, readonly, destination, true),
    )
    push!(p.schedule, id)
    return ValueRef(value)
end

function bind!(p, name::Symbol, value)
    # A source-level alias shares its value ID. Bind operations preserve Julia
    # scope and reflection, but are not consumers for producer-elimination.
    if haskey(p.declared_types, name)
        type = p.declared_types[name]
        term = :(Core.typeassert(Base.convert($type, $value), $type))
        value = operation!(p, BorrowOp(), term; parents=operands(value))
    end
    value isa ValueRef || (value = operation!(p, BorrowOp(), value; readonly=true))
    p.bindings[name] = value
    p.policy isa VisibleScope && name in p.protected && (p.values[value.id].escapes = true)
    push!(
        p.operations,
        Operation(BindOp(), :($name = $value), [value.id], Int[],
            p.statement, true, 0, true),
    )
    push!(p.schedule, length(p.operations))
    return value
end

# Rebuild adjacency after a rewrite; IDs and operand positions never change.
# The schedule remains authoritative for evaluation order. Effect constraints
# make it explicit that ordinary topological sorting cannot reorder side effects.
function rebuild!(p)
    p.dependencies = SimpleDiGraph(length(p.operations))
    active = filter(i -> p.operations[i].active, p.schedule)
    last_effect = 0
    reads = Int[]
    for id in active
        op = p.operations[id]
        op.inputs = operands(op.expression)
        for value in op.inputs
            producer = p.values[value].producer
            producer != 0 && p.operations[producer].active && producer != id &&
                add_edge!(p.dependencies, producer, id)
        end
        last_effect != 0 && add_edge!(p.dependencies, last_effect, id)
        if op.readonly
            push!(reads, id)
        else
            for before in reads
                add_edge!(p.dependencies, before, id)
            end
            empty!(reads)
            last_effect = id
        end
    end
    return p
end

include("graph_parse.jl")
include("graph_fusion.jl")
include("graph_emit.jl")

end
