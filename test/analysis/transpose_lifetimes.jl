using Test, LinearAlgebra, cuNumeric

_remember_transpose_parent!(ref, a) = (ref[] = a; nothing)

@testset "Lazy matrix wrapper lifetimes in @accelerate" begin
    cuNumeric.allowscalar(false)
    ah = ComplexF64[1+im 2-im; 3+2im 4+im]
    bh = ComplexF64[2-im 1+im; -1+im 3-2im]
    a, b = NDArray(ah), NDArray(bh)
    owner = Ref{Any}()

    @accelerate function _transpose_temporary(a, b, owner)
        t = copy(a)
        _remember_transpose_parent!(owner, t)
        w = transpose(t)
        w * b
    end
    @test Array(_transpose_temporary(a, b, owner)) ≈ transpose(ah) * bh
    @test owner[].ptr == C_NULL

    @accelerate function _transpose_vector_temporary(a, x, owner)
        t = copy(a)
        _remember_transpose_parent!(owner, t)
        w = transpose(t)
        w * x
    end
    @test Array(_transpose_vector_temporary(a, NDArray(bh[:, 1]), owner)) ≈ transpose(ah) * bh[:, 1]
    @test owner[].ptr == C_NULL

    @accelerate function _double_transpose_temporary(a, b, owner)
        t = copy(a)
        _remember_transpose_parent!(owner, t)
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
