using Test, LinearAlgebra, cuNumeric

struct _LifetimeBox{T}
    value::T
end
cuNumeric._lifetime_dependencies(x::_LifetimeBox) = (x.value,)
_lifetime_read(x::_LifetimeBox) = _lifetime_read(x.value)
_lifetime_read(x::Union{Transpose,Adjoint}) = _lifetime_read(parent(x))
_lifetime_read(x::NDArray) = Array(x)
_lifetime_remember!(ref, x) = (ref[] = x; nothing)
_lifetime_freed(ref) = ref[].ptr == C_NULL
_lifetime_pair(x) = (left=x, right=_LifetimeBox(x))

@testset "Type-dispatched lifetime dependencies" begin
    host = [1.0 2.0; 3.0 4.0]
    input = NDArray(host)
    owner = Ref{Any}()

    # The macro sees only an arbitrary callable, including nested dependencies
    # and an opaque function returning the very same NDArray.
    @accelerate function _dependency_consume(a, wrap, owner)
        t = copy(a)
        _lifetime_remember!(owner, t)
        w = wrap(t)
        result = _lifetime_read(w)
        freed = _lifetime_freed(owner)
        (result, freed)
    end
    for wrap in (identity, transpose, adjoint, _LifetimeBox,
                 x -> _LifetimeBox(transpose(x)))
        result, freed = _dependency_consume(input, wrap, owner)
        @test result == host
        @test freed  # Released before the function returns, after w's last use.
    end

    @accelerate function _dependency_escape(a, wrap)
        t = copy(a)
        wrap(t)
    end
    for wrap in (transpose, adjoint, _LifetimeBox)
        escaped = _dependency_escape(input, wrap)
        @test _lifetime_read(escaped) == host
        handle = only(cuNumeric._lifetime_handles(escaped))
        cuNumeric.destroy!(handle)
    end
    escaped = _dependency_escape(input, _lifetime_pair)
    @test Array(escaped.left) == host
    @test _lifetime_read(escaped.right) == host
    cuNumeric.destroy!(escaped.left)

    @accelerate function _dependency_input(w)
        t = parent(w)
        _lifetime_read(t)
    end
    @test _dependency_input(transpose(input)) == host
    @test Array(input) == host

    @accelerate function _dependency_siblings(a, owner)
        t = copy(a)
        _lifetime_remember!(owner, t)
        pair = _lifetime_pair(t)
        first = _lifetime_read(pair.left)
        second = _lifetime_read(pair.right)
        (first, second, _lifetime_freed(owner))
    end
    @test _dependency_siblings(input, owner) == (host, host, true)

    # Captured inputs, not just function arguments, are caller-owned.
    captured = input
    @accelerate function _dependency_capture()
        alias = identity(captured)
        _lifetime_read(alias)
    end
    @test _dependency_capture() == host
    @test Array(input) == host

    @accelerate function _dependency_rebind(a, owner)
        t = copy(a)
        _lifetime_remember!(owner, t)
        t = transpose(t)
        alias = _LifetimeBox(t)
        t = nothing
        result = _lifetime_read(alias)
        (result, _lifetime_freed(owner))
    end
    @test _dependency_rebind(input, owner) == (host, true)

    @accelerate function _dependency_swap(a, owner)
        x = copy(a)
        _lifetime_remember!(owner, x)
        y = _LifetimeBox(x)
        x, y = y, x
        first = _lifetime_read(x)
        second = _lifetime_read(y)
        (first, second, _lifetime_freed(owner))
    end
    @test _dependency_swap(input, owner) == (host, host, true)

    # A callable named transpose can allocate an independent handle. It must
    # not inherit the input's protection merely because of its spelling.
    @accelerate function _dependency_independent(a, transpose, owner)
        t = transpose(a)
        _lifetime_remember!(owner, t)
        result = _lifetime_read(t)
        (result, _lifetime_freed(owner))
    end
    @test _dependency_independent(input, copy, owner) == (host, true)
    @test _dependency_independent(input, cuNumeric.transpose, owner) == (permutedims(host), true)
    @test Array(input) == host

    @accelerate function _dependency_scalar(a, owner)
        t = copy(a)
        _lifetime_remember!(owner, t)
        scalar = cuNumeric.cnscalar(t)
        result = Array(scalar.value)
        (result, _lifetime_freed(owner))
    end
    scalar_input = NDArray(fill(3.0))
    @test _dependency_scalar(scalar_input, owner) == (fill(3.0), true)
    @test Array(scalar_input) == fill(3.0)

    # Both soft-scope named bindings survive; hard-scope temporaries do not.
    @accelerate begin
        soft_parent = copy(input)
        soft_wrapper = _LifetimeBox(soft_parent)
        nothing
    end
    @test Array(soft_parent) == host
    @test _lifetime_read(soft_wrapper) == host
    escaped = @accelerate let
        t = copy(input)
        _LifetimeBox(t)
    end
    @test _lifetime_read(escaped) == host
    @test !(@isdefined t)

    # A wrapper with no locally-bound owner leaves ownership with the GC.
    hidden_owner = _dependency_escape(input, x -> _LifetimeBox(copy(x)))
    @test _lifetime_read(hidden_owner) == host
    cuNumeric.destroy!(hidden_owner.value)
    cuNumeric.destroy!(escaped.value)
    cuNumeric.destroy!(soft_parent)
    cuNumeric.destroy!(scalar_input)
    cuNumeric.destroy!(input)
end
