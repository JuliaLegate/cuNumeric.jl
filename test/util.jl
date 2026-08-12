#= Copyright 2026 Northwestern University,
 *                   Carnegie Mellon University University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author(s): David Krasowska <krasow@u.northwestern.edu>
 *            Ethan Meitz <emeitz@andrew.cmu.edu>
=#

# Domain-aware test data generation

const DOMAIN_GENERATORS = Dict{Symbol,Function}(
    :greater_than_one => (T, N) -> abs.(rand(T, N)) .+ one(T),
    :unit_interval => (T, N) -> T(2) .* rand(T, N) .- one(T),
    :normal => (T, N) -> randn(T, N),
    :uniform => (T, N) -> rand(T, N),
    :positive => (T, N) -> (x=rand(T, N); T <: Signed ? abs.(max.(x, -typemax(T))) : x),
)

rtol(::Type{Float16}) = 1e-2
rtol(::Type{Float32}) = 1e-5
rtol(::Type{Float64}) = 1e-12
rtol(::Type{I}) where {I<:Integer} = rtol(float(I))
atol(::Type{Float16}) = 1e-3
atol(::Type{Float32}) = 1e-8
atol(::Type{Float64}) = 1e-15
atol(::Type{I}) where {I<:Integer} = atol(float(I))
rtol(::Type{Complex{T}}) where {T} = rtol(T)
atol(::Type{Complex{T}}) where {T} = atol(T)

# Reduction rounding error grows with the number of elements reduced (n).
reduction_rtol(::Type{T}, n) where {T} = rtol(T) * n
reduction_atol(::Type{T}, n) where {T} = atol(T) * n

is_same(arr1::NDArray, arr2::NDArray) = @allowscalar (arr1 == arr2)[1]
is_same(arr1::NDArray, arr2::Array) = @allowscalar (arr1 == arr2)[1]
is_same(arr1::Array, arr2::NDArray) = @allowscalar (arr1 == arr2)[1]
is_same(arr1::Array, arr2::Array) = (arr1 == arr2)

function my_rand(::Type{F}, dims...; L=F(-1000), R=F(1000)) where {F<:AbstractFloat}
    return L .+ (R-L) .* rand(F, dims...)
end
function my_rand(::Type{I}, dims...; L=nothing, R=nothing) where {I<:Integer}
    L_default = I <: Unsigned ? 0 : max(-255, Int64(typemin(I)))
    R_default = I <: Unsigned ? 255 : min(255, Int64(typemax(I)))
    L_val = isnothing(L) ? I(L_default) : I(L)
    R_val = isnothing(R) ? I(R_default) : I(R)
    res = Float64(L_val) .+ floor.((Float64(R_val) - Float64(L_val) + 1.0) .* rand(dims...))
    return floor.(I, res)
end
function my_rand(::Type{CT}, dims...; L=T(-100), R=T(100)) where {T,CT<:Complex{T}}
    return Complex.(my_rand(T, dims...; L=L, R=R), my_rand(T, dims...; L=L, R=R))
end
my_rand(::Type{Bool}, dims...) = rand(Bool, dims...)

function make_julia_arrays(T, N, domain_key::Symbol; count::Int=1)
    generator = DOMAIN_GENERATORS[domain_key]

    julia_arrs_1D = [generator(T, N) for _ in 1:count]
    julia_arrs_2D = [reshape(generator(T, N), (isqrt(N), isqrt(N))) for _ in 1:count]

    return julia_arrs_1D..., julia_arrs_2D...
end

function make_cunumeric_arrays(julia_arrs_1D, julia_arrs_2D, T, N; count::Int=1)
    N_2D_dims = (isqrt(N), isqrt(N))

    cunumeric_arrs = [cuNumeric.zeros(T, N) for _ in 1:count]
    cunumeric_arrs_2D = [cuNumeric.zeros(T, N_2D_dims...) for _ in 1:count]

    @allowscalar begin
        for c in 1:count
            for i in 1:N
                cunumeric_arrs[c][i] = julia_arrs_1D[c][i]
            end
            for i in 1:N_2D_dims[1], j in 1:N_2D_dims[2]
                cunumeric_arrs_2D[c][i, j] = julia_arrs_2D[c][i, j]
            end
        end
    end

    return cunumeric_arrs..., cunumeric_arrs_2D...
end

function safe_isapprox(x, y, atol, rtol)
    # Handle NaN
    if isnan(x) && isnan(y)
        return true
    end

    # Handle Inf (must be same sign)
    if isinf(x) && isinf(y)
        return x === y
    end

    # Handle mixed finite/non-finite
    if isfinite(x) != isfinite(y)
        return false
    end

    return isapprox(x, y; atol=atol, rtol=rtol)
end

function _safe_compare_fail(idx, left, right, atol, rtol)
    absdiff = abs(left - right)
    scale = max(abs(left), abs(right))
    tol = max(atol, rtol * scale)
    @error "safe_compare mismatch" index=idx left=left right=right absdiff=absdiff atol=atol rtol=rtol tol=tol
    return false
end

# Argument order matches cuNumeric.compare(..., atol, rtol) and existing call sites.
function safe_compare(
    julia_array::AbstractArray{T1,N}, arr::NDArray{T2,N}, atol, rtol
) where {T1,T2,N}
    if cuNumeric.shape(arr) != size(julia_array)
        @error "safe_compare shape mismatch" left_size=size(julia_array) right_shape=cuNumeric.shape(
            arr
        )
        return false
    end

    for CI in CartesianIndices(julia_array)
        x = julia_array[CI]
        y = arr[Tuple(CI)...]
        if !safe_isapprox(x, y, atol, rtol)
            return _safe_compare_fail(Tuple(CI), x, y, atol, rtol)
        end
    end

    return true
end

function safe_compare(
    arr::NDArray{T2,N}, julia_array::AbstractArray{T1,N}, atol, rtol
) where {T1,T2,N}
    return safe_compare(julia_array, arr, atol, rtol)
end

function safe_compare(
    arr::NDArray{T1,N}, arr2::NDArray{T2,N}, atol, rtol
) where {T1,T2,N}
    if cuNumeric.shape(arr) != cuNumeric.shape(arr2)
        @error "safe_compare shape mismatch" left_shape=cuNumeric.shape(arr) right_shape=cuNumeric.shape(
            arr2
        )
        return false
    end

    for CI in CartesianIndices(cuNumeric.shape(arr))
        x = arr[Tuple(CI)...]
        y = arr2[Tuple(CI)...]
        if !safe_isapprox(x, y, atol, rtol)
            return _safe_compare_fail(Tuple(CI), x, y, atol, rtol)
        end
    end

    return true
end
