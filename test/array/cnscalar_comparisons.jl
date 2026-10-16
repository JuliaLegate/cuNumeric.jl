using Test, cuNumeric

function comparison_outcome(f)
    try
        return f()
    catch err
        return typeof(err)
    end
end

@testset "CNScalar derived comparisons" begin
    # Include all wrapper families, unordered values, signed zero, and mixed
    # signed/unsigned inputs. Compare with the previous direct backend operation
    # as well as checking the fetch permission on every argument arrangement.
    real_values = (false, true, Int32(-1), typemin(Int64), UInt32(2),
                   typemax(UInt64), -0.0f0, 0.0, 2.5, Inf, NaN)
    complex_values = (ComplexF32(1, 2), ComplexF64(NaN, 0))
    values = (real_values..., complex_values...)
    wrapped = map(x -> cnscalar(NDArray(x)), values)
    allowfetch(false) do
        cuNumeric.allowpromotion() do
            for (i, x) in pairs(wrapped), (j, y) in pairs(wrapped)
                ops = i <= length(real_values) && j <= length(real_values) ?
                      (!=, >, >=) : (!=,)
                for op in ops, (left, right) in
                    ((x, y), (x, values[j]), (values[i], y))
                    @test_throws ArgumentError op(left, right)
                    allowfetch() do
                        actual = comparison_outcome(() -> op(left, right))
                        previous = comparison_outcome(() -> cuNumeric._scalar_compare(op, left, right))
                        @test actual === previous
                    end
                    @test_throws ArgumentError op(left, right)
                end
            end
            # Native values of the same type also provide an independent oracle.
            for (value, x) in zip(values, wrapped)
                ops = value isa Real ? (!=, >, >=) : (!=,)
                for op in ops
                    @test (@allowfetch op(x, x)) === op(value, value)
                    @test (@allowfetch op(x, value)) === op(value, value)
                    @test (@allowfetch op(value, x)) === op(value, value)
                end
            end
        end
    end
    foreach(cuNumeric.destroy!, wrapped)
end
