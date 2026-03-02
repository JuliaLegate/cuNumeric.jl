@testset "Stability" begin

    @testset "core" begin
        a = cuNumeric.zeros(5)
        b = cuNumeric.zeros(Float64, 3, 4)
        @inferred size(a)
        @inferred size(b)
        @inferred cuNumeric.shape(a)
        @inferred cuNumeric.shape(b)
    end

    @testset "construction" begin
        # zeros, zeros_like, ones, rand, fill, trues, falses\
        for constructor in (:zeros, :ones)
            @eval begin
                @inferred cuNumeric.$(constructor)(Float64, 3, 2)
                @inferred cuNumeric.$(constructor)(Float64, (3, 4))
                @inferred cuNumeric.$(constructor)(3, 5, 6)
                @inferred cuNumeric.$(constructor)((3,))
                @inferred cuNumeric.$(constructor)()
                @inferred cuNumeric.$(constructor)(Int64)
            end
        end
        a = cuNumeric.zeros(Float64, 5, 3)
        @inferred cuNumeric.zeros_like(a)

        for constructor in (:trues, :falses)
            @eval begin
                @inferred cuNumeric.$(constructor)(5)
                @inferred cuNumeric.$(constructor)((5,4))
                @inferred cuNumeric.$(constructor)(3, 4, 5)
            end
        end

        @inferred cuNumeric.fill(2.0, 3, 4)
        @inferred cuNumeric.fill(2, (3, 4))
        @inferred cuNumeric.fill(2.0, 3)

        @inferred cuNumeric.rand(4, 3)
        @inferred cuNumeric.rand(Float32, 5)
    end

    @testset "conversion" begin
        # cast to array, as_type
        a = cuNumeric.zeros(Float64, 5, 5)
        @inferred Array(a)
        @inferred Array{Float32}(a)
        @inferred cuNumeric.as_type(a, Float32)
        @inferred cuNumeric.as_type(a, Int64)
    end

    @testset "indexing" begin
        # getindex, setindex!, copy, copyto!, fill!, as_type
        a = cuNumeric.zeros(Float32, 5, 5)
        b = cuNumeric.zeros(Int32, 11)

        @inferred a[1:3, 1:3]
        @inferred a[2, 1:3]
        @inferred a[1, 1:3] .+ b[1:3]
        @inferred b[1:5]
        # @inferred a[1:3, 1:end]
        allowscalar() do
            @inferred a[1, 2]
        end
    end

    @testset "broadcasting" begin
        a = cuNumeric.ones(Float32, 3, 3)
        b = cuNumeric.ones(Int32, 3, 3)
        @inferred 5 .* a
        @inferred 5.0f0 .* a
        @inferred 5 * a
        @inferred 5.0f0 * a

        @inferred a .* b
        @inferred a .+ b
        @inferred a ./ b
        @inferred ((a .* b) .+ a) .* 2.0f0
    end


end