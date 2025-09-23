using Pkg
Pkg.activate("../cn_dev")
using cuNumeric
include("/pool/emeitz/repos/cuNumeric.jl/test/tests/util.jl")

N = 9

get_pwrs(::Type{I}) where {I<:Integer} = I.([-10, -5, -2, -1, 0, 1, 2, 5, 10])
get_pwrs(::Type{F}) where {F<:AbstractFloat} = F.([-3.141, -2, -1, 0, 1, 2, 3.2, 4.41, 6.233])
get_pwrs(::Type{Bool}) = [true, false, true, false, false, true, false, true, true]

function test_pair(BT, PT)
    base_jl = my_rand(BT, N)

    if BT <: Union{Bool,Int32} && PT == Int32
        # julia doesnt like Int32 powers
        pwrs = Float32.(get_pwrs(PT))
    elseif BT <: Union{Bool,Int32,Int64} && PT <: Union{Int32,Int64}
        # julia doesnt like Int64 powers on bool
        pwrs = Float64.(get_pwrs(PT))
    elseif (PT <: AbstractFloat) && (BT <: AbstractFloat || BT <: Signed)
        # Things like -387 ^ 3.2 will be Complex and error
        pwrs = get_pwrs(PT)
        base_jl = abs.(base_jl)
    else
        pwrs = get_pwrs(PT)
    end

    base_cn = @allowscalar NDArray(base_jl)
    pwrs_cn = @allowscalar NDArray(pwrs)

    # we deviate a bit form Julia here
    if (PT <: Union{Int32,Int64}) && (BT <: Union{Bool,Int32,Int64})
        T_OUT = cuNumeric.__my_promote_type(typeof(^), BT, PT)
    else
        T_OUT = Base.promote_op(Base.:(^), BT, PT)
    end

    allowpromotion(sizeof(BT) != sizeof(PT)) do
        # allowpromotion(PT == Bool || PT == Int32 || BT == Bool) do #sizeof(BT) != sizeof(PT)
        allowscalar() do
            # Power is array
            res = cuNumeric.compare(base_jl .^ pwrs, base_cn .^ pwrs_cn, atol(T_OUT), rtol(T_OUT))

            res || println(
                "Failed first. \nExpected: $(base_jl .^ pwrs). \nGot $((base_cn .^ pwrs_cn)[:])"
            )

            res = true

            # Power is scalar
            for p in pwrs
                res |= cuNumeric.compare(base_jl .^ p, base_cn .^ p, atol(T_OUT), rtol(T_OUT))
            end

            res || println("Failed second")
        end
    end
end

TYPES = Base.uniontypes(cuNumeric.SUPPORTED_TYPES)
for (BT, PT) in Iterators.product(TYPES, TYPES)
    println("$(BT) ^ $(PT)")
    test_pair(BT, PT)
end
