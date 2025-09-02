
# Map functions to their required domains
const SPECIAL_DOMAINS = Dict(
    Base.acosh => :greater_than_one,
    Base.log => :positive,
    Base.log10 => :positive,
    Base.log2 => :positive,
    Base.log1p => :positive, # technicaly anything > -1
    Base.sqrt => :positive,
)


function test_unary_operation(func, julia_arr, julia_arr_2D, cunumeric_arr, cunumeric_arr_2D, T)
    
    T_OUT = Base.promote_op(func, T)
    
    # Pre-allocate output arrays
    cunumeric_in_place = cuNumeric.zeros(T_OUT, length(julia_arr))
    cunumeric_in_place_2D = cuNumeric.zeros(T_OUT, size(julia_arr_2D)...)
    
    # Compute results using different methods
    julia_res = func.(julia_arr)
    julia_res_2D = func.(julia_arr_2D)
    
    cunumeric_res = func.(cunumeric_arr)
    cunumeric_res_2D = func.(cunumeric_arr_2D)
    cunumeric_in_place .= func.(cunumeric_arr)
    cunumeric_in_place_2D .= func.(cunumeric_arr_2D)
    cunumeric_res2 = map(func, cunumeric_arr)
    
    allowscalar() do
        @test cuNumeric.compare(julia_res, cunumeric_in_place, atol(T_OUT), rtol(T_OUT))
        @test cuNumeric.compare(julia_res, cunumeric_res, atol(T_OUT), rtol(T_OUT))
        @test cuNumeric.compare(julia_res, cunumeric_res2, atol(T_OUT), rtol(T_OUT))
        @test cuNumeric.compare(julia_res_2D, cunumeric_res_2D, atol(T_OUT), rtol(T_OUT))
        @test cuNumeric.compare(julia_res_2D, cunumeric_in_place_2D, atol(T_OUT), rtol(T_OUT))
    end
end

function test_unary_function_set(func_dict, T, N)

    default_generator = (T == Bool) ? :uniform : :unit_interval

    skip_on_integer = (Base.atanh, Base.atan, Base.acos, Base.asin)
    skip_on_bool = (Base.:(-), skip_on_integer...)
    
    @testset "$func" for func in keys(func_dict)

        # The are only defined for like 3 integers (-1, 0, 1) so just skip them
        if func in skip_on_integer && (T <: Integer)
            continue
        end

        if func in skip_on_bool && (T == Bool)
            continue
        end

        domain_type = get(SPECIAL_DOMAINS, func, default_generator)

        # :uniform is the only generator capable of generating bits
        skip = (T == Bool && domain_type != :uniform)
        skip && continue

        julia_arr_1D, julia_arr_2D = make_julia_arrays(T, N, domain_type)
        cunumeric_arr, cunumeric_arr_2D = make_cunumeric_arrays(julia_arr_1D, julia_arr_2D, T)
        
        test_unary_operation(func, julia_arr_1D, julia_arr_2D, 
                                cunumeric_arr, cunumeric_arr_2D, T)
    end
end