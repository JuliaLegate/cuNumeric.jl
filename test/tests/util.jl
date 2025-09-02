# Domain-aware test data generation


const DOMAIN_GENERATORS = Dict{Symbol, Function}(
    :greater_than_one => (T, N) -> abs.(rand(T, N)) .+ one(T),
    :unit_interval => (T, N) -> T(2) .* rand(T, N) .- one(T),
    :normal => (T, N) -> randn(T, N),
    :uniform => (T, N) -> rand(T, N),
    :positive => (T, N) -> abs.(rand(T,N))
)


function make_julia_arrays(T, N, domain_key::Symbol)
    generator = DOMAIN_GENERATORS[domain_key]
    
    julia_arr_1D = generator(T, N)
    julia_arr_2D = reshape(generator(T, N), (isqrt(N), isqrt(N)))
    
    return julia_arr_1D, julia_arr_2D
end

function make_cunumeric_arrays(julia_arr_1D, julia_arr_2D, T)
    N_1D = length(julia_arr_1D)
    N_2D_dims = size(julia_arr_2D)
    
    cunumeric_arr = cuNumeric.zeros(T, N_1D)
    cunumeric_arr_2D = cuNumeric.zeros(T, N_2D_dims...)
    
    @allowscalar begin
        for i in 1:N_1D
            cunumeric_arr[i] = julia_arr_1D[i]
        end
        for i in 1:N_2D_dims[1], j in 1:N_2D_dims[2]
            cunumeric_arr_2D[i,j] = julia_arr_2D[i,j]
        end
    end
    
    return cunumeric_arr, cunumeric_arr_2D
end