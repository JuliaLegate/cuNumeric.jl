using Test, LinearAlgebra, cuNumeric
using InteractiveUtils: code_typed
using StructArrays

struct _LifetimeBox{T}
    value::T
end
struct _LifetimePair{A,B}
    left::A
    right::B
end
cuNumeric._lifetime_dependencies(::Type{<:_LifetimeBox}) = cuNumeric.DependsOn((1,))
cuNumeric._lifetime_dependencies(::Type{<:_LifetimePair}) = cuNumeric.DependsOnAll()
_lifetime_read(x::_LifetimeBox) = _lifetime_read(x.value)
_lifetime_read(x::Union{Transpose,Adjoint}) = _lifetime_read(parent(x))
_lifetime_read(x::NDArray) = Array(x)
_lifetime_read(x::_LifetimePair) = (_lifetime_read(x.left), _lifetime_read(x.right))
_lifetime_remember!(ref, x) = (ref[]=x; nothing)
_lifetime_freed(ref) = ref[].ptr == C_NULL
# These test helpers return independent host values; they do not borrow inputs.
cuNumeric._lifetime_dependencies(::typeof(_lifetime_read)) = cuNumeric.Independent()
cuNumeric._lifetime_dependencies(::typeof(_lifetime_remember!)) = cuNumeric.Independent()
cuNumeric._lifetime_dependencies(::typeof(_lifetime_freed)) = cuNumeric.Independent()

@accelerate function _static_consume(a, owner)
    t = copy(a)
    _lifetime_remember!(owner, t)
    w = _LifetimeBox(transpose(t))
    result = _lifetime_read(w)
    (result, _lifetime_freed(owner))
end
@accelerate function _static_pair(a, b, owner_a, owner_b)
    x = copy(a)
    y = copy(b)
    _lifetime_remember!(owner_a, x)
    _lifetime_remember!(owner_b, y)
    pair = _LifetimePair(transpose(x), _LifetimeBox(y))
    result = _lifetime_read(pair)
    (result, _lifetime_freed(owner_a), _lifetime_freed(owner_b))
end
@accelerate function _static_escape(a)
    t = copy(a)
    _LifetimeBox(transpose(t))
end
@accelerate function _static_pair_escape(a, b)
    x = copy(a)
    y = copy(b)
    args = (x, y)
    _LifetimePair(args...)
end
@accelerate function _static_rebind(a, owner)
    t = copy(a)
    _lifetime_remember!(owner, t)
    t = transpose(t)
    alias = _LifetimeBox(t)
    t = nothing
    result = _lifetime_read(alias)
    (result, _lifetime_freed(owner))
end
@accelerate function _static_swap(a, owner)
    x = copy(a)
    _lifetime_remember!(owner, x)
    y = _LifetimeBox(x)
    x, y = y, x
    first_result = _lifetime_read(x)
    second_result = _lifetime_read(y)
    (first_result, second_result, _lifetime_freed(owner))
end
@accelerate function _static_parent(w)
    x = parent(w)
    _lifetime_read(x)
end
_lifetime_keyword_alias(x; ignored=0) = x
cuNumeric._lifetime_dependencies(::typeof(_lifetime_keyword_alias)) = cuNumeric.DependsOn((1,))
@accelerate function _static_keyword(a)
    w = _lifetime_keyword_alias(a; ignored=2)
    Array(w)
end
@accelerate function _static_splat(a)
    args = (a,)
    w = identity(args...)
    Array(w)
end
@accelerate function _static_replace(a, owner)
    x = copy(a)
    _lifetime_remember!(owner, x)
    w = transpose(x)
    x = copy(a)
    result = (_lifetime_read(w), Array(x))
    (result, _lifetime_freed(owner))
end
@accelerate function _static_destructure(a, b, owner_a, owner_b)
    x = copy(a)
    y = copy(b)
    _lifetime_remember!(owner_a, x)
    _lifetime_remember!(owner_b, y)
    pair = (x, _LifetimeBox(y))
    first_array, last_wrapper = pair
    result = (_lifetime_read(first_array), _lifetime_read(last_wrapper))
    (result, _lifetime_freed(owner_a), _lifetime_freed(owner_b))
end

@testset "Static ownership dependencies" begin
    host = [1.0 2.0; 3.0 4.0]
    # Use backend-owned inputs so the typed observation refs also accept copies.
    input = copy(NDArray(host))
    owner, second_owner = Ref(input), Ref(input)
    @test (@inferred _static_consume(input, owner)) == (host, true)
    @test _static_pair(input, input, owner, second_owner) == ((host, host), true, true)
    @test (@inferred _static_rebind(input, owner)) == (host, true)
    @test (@inferred _static_swap(input, owner)) == (host, host, true)
    @test _static_parent(transpose(input)) == host
    @test _static_keyword(input) == host
    @test _static_splat(input) == host
    @test (@inferred _static_replace(input, owner)) == ((host, host), true)
    @test _static_destructure(input, input, owner, second_owner) == ((host, host), true, true)
    escaped = _static_escape(input)
    @test _lifetime_read(escaped) == host
    cuNumeric.destroy!(parent(escaped.value))
    escaped_pair = _static_pair_escape(input, input)
    @test _lifetime_read(escaped_pair) == (host, host)
    cuNumeric.destroy!(escaped_pair.left)
    cuNumeric.destroy!(escaped_pair.right)
    @test Array(input) == host

    # Soft-scope bindings survive; returning a wrapper from a let protects its owner.
    @accelerate begin
        soft_parent = copy(input)
        soft_wrapper = _LifetimeBox(soft_parent)
        nothing
    end
    @test _lifetime_read(soft_wrapper) == host
    escaped = @accelerate let
        t = copy(input)
        _LifetimeBox(t)
    end
    @test _lifetime_read(escaped) == host
    @test !(@isdefined t)
    cuNumeric.destroy!(escaped.value)
    cuNumeric.destroy!(soft_parent)
    cuNumeric.destroy!(input)
end

# No rule: the analyzer must not inspect these functions' bodies.
_lifetime_unknown_copy(a) = copy(a)
_lifetime_unknown_wrap(a) = _LifetimeBox(a)
@accelerate function _unknown_result(a, owner)
    t = _lifetime_unknown_copy(a)
    _lifetime_remember!(owner, t)
    result = Array(t)
    (result, _lifetime_freed(owner))
end
@accelerate function _unknown_wrapper(a, owner)
    t = copy(a)
    _lifetime_remember!(owner, t)
    w = _lifetime_unknown_wrap(t)
    result = _lifetime_read(w)
    (result, _lifetime_freed(owner))
end
@accelerate function _unknown_escape(a)
    t = copy(a)
    _lifetime_unknown_wrap(t)
end
@accelerate function _shadowed_transpose(a, transpose, owner)
    t = transpose(a)
    _lifetime_remember!(owner, t)
    result = Array(t)
    (result, _lifetime_freed(owner))
end
@accelerate function _unknown_rebind(a, first_owner, second_owner)
    x = _lifetime_unknown_copy(a)
    _lifetime_remember!(first_owner, x)
    x = _lifetime_unknown_copy(a)
    _lifetime_remember!(second_owner, x)
    result = Array(x)
    (result, _lifetime_freed(first_owner), _lifetime_freed(second_owner))
end
@testset "Unknown calls wait until scope exit" begin
    host = [1.0 2.0; 3.0 4.0]
    # Use backend-owned inputs so the typed observation refs also accept copies.
    input = copy(NDArray(host))
    owner = Ref(input)
    @test (@inferred _unknown_result(input, owner)) == (host, false)
    @test _lifetime_freed(owner)
    second_owner = Ref(input)
    @test _unknown_rebind(input, owner, second_owner) == (host, false, false)
    @test _lifetime_freed(owner) && _lifetime_freed(second_owner)
    @test _unknown_wrapper(input, owner) == (host, false)
    @test _lifetime_freed(owner)
    @test (@inferred _shadowed_transpose(input, copy, owner)) == (host, false)
    @test _lifetime_freed(owner)
    escaped = _unknown_escape(input)
    @test _lifetime_read(escaped) == host
    cuNumeric.destroy!(escaped.value)
    @test Array(input) == host
    cuNumeric.destroy!(input)
end

@testset "Constructor rules and expansion" begin
    selected = @inferred cuNumeric.DependsOn((1, 2))
    @test selected.positions == (1, 2)
    @test typeof(selected) === typeof(cuNumeric.DependsOn((2, 3)))
    @test isconcretetype(fieldtype(typeof(selected), :positions))
    # SVD is also a backend task ID in cuNumeric. Always use the LA type here.
    @test cuNumeric._lifetime_dependencies(LinearAlgebra.SVD).positions == (1, 2, 3)
    @test cuNumeric._lifetime_dependencies(LinearAlgebra.Eigen).positions == (1, 2)
    @test cuNumeric._lifetime_dependencies(LinearAlgebra.Cholesky).positions == (1,)
    @test cuNumeric._lifetime_dependencies(cuNumeric.NDArrayQR).positions == (1, 2)
    @test cuNumeric._lifetime_dependencies(StructArray).positions == (1,)
    @test cuNumeric._lifetime_dependencies(_lifetime_unknown_copy) isa cuNumeric.UnknownDependencies
    @test cuNumeric._lifetime_dependencies(*, Val(1)).positions == (1,)
    @test cuNumeric._lifetime_dependencies(*, Val(2)) isa cuNumeric.Independent
    @test cuNumeric._lifetime_dependencies(copyto!).positions == (1,)
    expanded = cuNumeric._accelerate_expand(:(function f(a)
            x = copy(a)
            w = transpose(x)
            y = Array(w)
            y
        end), @__MODULE__)
    text = string(expanded)
    @test !occursin("_LifetimeScope", text)
    @test !occursin("_lifetime_", text)
    @test !occursin("Dict", text)
    @test occursin("maybe_insert_delete", text)

    # Inserting cleanup must not obscure adjacent updates from sibling fusion.
    source = quote
        u_new[2:(end - 1), 2:(end - 1)] .= u[2:(end - 1), 2:(end - 1)] .+ 1
        v_new[2:(end - 1), 2:(end - 1)] .= v[2:(end - 1), 2:(end - 1)] .+ 2
        nothing
    end
    fused = cuNumeric.AccelerateGraph.compile(source; aggressive=true,
        protected=Set([:u_new, :v_new, :u, :v]), caller=@__MODULE__)
    @test occursin("copyto_fused_siblings!", string(fused))
end

# Isolate the generated ownership code from backend compilation. The same
# cleanup dispatch applies to these typed tokens, with no runtime or GPU needed.
mutable struct _LifetimeToken{T}
    value::T
    deleted::Bool
end
Base.copy(x::_LifetimeToken) = _LifetimeToken(x.value, false)
cuNumeric.maybe_insert_delete(x::_LifetimeToken) = (x.deleted=true; nothing)
_lifetime_value(x::_LifetimeBox) = _lifetime_value(x.value)
_lifetime_value(x::_LifetimeToken) = (x.deleted && error("premature cleanup"); x.value)
cuNumeric._lifetime_dependencies(::typeof(_lifetime_value)) = cuNumeric.Independent()
@accelerate function _typed_lifetime(a)
    x = copy(a)
    w = _LifetimeBox(_LifetimeBox(x))
    result = _lifetime_value(w)
    result
end
@accelerate function _scalar_lifetime_cleanup(x::Float64)
    y = x + 1.0
    y * 2.0
end
_cleanup_allocations(x) = @allocated _scalar_lifetime_cleanup(x)
@testset "Generated lifetime code preserves concrete types" begin
    _cleanup_allocations(3.0)
    @test _cleanup_allocations(3.0) == 0
    for value in (1.0, Int32(3), ComplexF64(2, 3))
        token = _LifetimeToken(value, false)
        @test (@inferred _typed_lifetime(token)) === value
        info, result = only(code_typed(_typed_lifetime, (typeof(token),); optimize=true))
        @test result === typeof(value)
        # Check intermediate values, not just the function's return type. Control
        # flow nodes have no value and may carry Any in Julia's CodeInfo metadata.
        @test all(zip(info.code, info.ssavaluetypes)) do (node, type)
            !(node isa Expr || node isa Core.PhiNode || node isa Core.PiNode) ||
                type isa Core.Const || type === Union{} || isconcretetype(type)
        end
        @test !token.deleted
    end
end

@testset "Lazy matrix wrapper lifetimes in @accelerate" begin
    cuNumeric.allowscalar(false)
    ah = ComplexF64[1+im 2-im; 3+2im 4+im]
    bh = ComplexF64[2-im 1+im; -1+im 3-2im]
    a, b = NDArray(ah), NDArray(bh)
    owner = Ref{Any}()

    @accelerate function _transpose_temporary(a, b, owner)
        t = copy(a)
        _lifetime_remember!(owner, t)
        w = transpose(t)
        w * b
    end
    @test Array(_transpose_temporary(a, b, owner)) ≈ transpose(ah) * bh
    @test owner[].ptr == C_NULL

    @accelerate function _transpose_vector_temporary(a, x, owner)
        t = copy(a)
        _lifetime_remember!(owner, t)
        w = transpose(t)
        w * x
    end
    @test Array(_transpose_vector_temporary(a, NDArray(bh[:, 1]), owner)) ≈ transpose(ah) * bh[:, 1]
    @test owner[].ptr == C_NULL

    @accelerate function _double_transpose_temporary(a, b, owner)
        t = copy(a)
        _lifetime_remember!(owner, t)
        w = LinearAlgebra.Transpose(t)
        u = Base.transpose(w)
        first_result = transpose(u) * b
        second_result = transpose(t) * b
        (first_result, second_result)
    end
    first_result, second_result = _double_transpose_temporary(a, b, owner)
    @test Array(first_result) ≈ transpose(ah) * bh
    @test Array(second_result) ≈ transpose(ah) * bh
    @test owner[].ptr == C_NULL

    @accelerate function _return_adjoint(a)
        t = copy(a)
        t'
    end
    escaped = _return_adjoint(a)
    @test Array(parent(escaped)) == ah
    @test_throws ArgumentError escaped * b
    cuNumeric.destroy!(parent(escaped))

    @accelerate function _return_transpose(a)
        t = copy(a)
        w = transpose(t)
        w
    end
    escaped = _return_transpose(a)
    @test Array(escaped * b) ≈ transpose(ah) * bh
    cuNumeric.destroy!(parent(escaped))
    @test Array(a) == ah
    @test Array(b) == bh
end
