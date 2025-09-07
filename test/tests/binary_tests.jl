
function test_binary_operation(func, julia_arr1, julia_arr2, cunumeric_arr1, cunumeric_arr2, T)
    
    T_OUT = Base.promote_op(func, T, T)
    
    # Pre-allocate output arrays
    cunumeric_in_place = cuNumeric.zeros(T_OUT, size(cunumeric_arr1)...)
    
    julia_res = func.(julia_arr1, julia_arr2)
    cunumeric_res = func.(cunumeric_arr1, cunumeric_arr2)
    cunumeric_in_place .= func.(cunumeric_arr1, cunumeric_arr2)
    cunumeric_res2 = map(func, cunumeric_arr1, cunumeric_arr2)
    
    allowscalar() do
        @test safe_compare(julia_res, cunumeric_in_place, atol(T_OUT), rtol(T_OUT))
        @test safe_compare(julia_res, cunumeric_res, atol(T_OUT), rtol(T_OUT))
        @test safe_compare(julia_res, cunumeric_res2, atol(T_OUT), rtol(T_OUT))
    end
end

function test_binary_function_set(func_dict, T, N)


    @testset "$func" for func in keys(func_dict)

        arrs_jl = make_julia_arrays(T, N, :uniform; count = 2)
        arrs_cunum = make_cunumeric_arrays(arrs_jl[1:2], arrs_jl[3:4], T, N; count = 2)
        
        test_binary_operation(func, arrs_jl[1:2]..., arrs_cunum[1:2]..., T)
        test_binary_operation(func, arrs_jl[3:4]..., arrs_cunum[3:4]..., T)
        
    end
end