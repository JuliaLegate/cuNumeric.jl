# Dimension guards
function solve(a::NDArray{T,1}, b::NDArray{S,M}) where {T,S,M}
    throw(ArgumentError("1-dimensional array given. Array must be at least two-dimensional"))
end

function solve(a::NDArray{T,0}, b::NDArray{S,M}) where {T,S,M}
    throw(ArgumentError("0-dimensional array given. Array must be at least two-dimensional"))
end

function solve(a::NDArray{T,N}, b::NDArray{S,0}) where {T,N,S}
    throw(ArgumentError("0-dimensional array given. Array must be at least one-dimensional"))
end

# 2D case: (m,m),(m)->( m)
function solve(a::NDArray{T,2}, b::NDArray{S,1}) where {T,S}
    size(a)[end - 1] != size(a)[end] &&
        throw(ArgumentError("Last 2 dimensions of the array must be square"))
    size(a)[2] != size(b)[1] &&
        throw(
            ArgumentError(
                "Input operand 1 has a mismatch in its dimension 0, " *
                "with signature (m,m),(m)->(m) (size $(size(b)[1]) " *
                "is different from $(size(a)[2]))",
            ),
        )
    prod(size(a)) == 0 || prod(size(b)) == 0 && return zeros(T, size(b)...)
    x = zeros(T, size(b)...)
    solve_batched(a, b, x)
    return x
end

# 2D case: (m,m),(m,n)->(m,n)
function solve(a::NDArray{T,2}, b::NDArray{S,2}) where {T,S}
    size(a)[end - 1] != size(a)[end] &&
        throw(ArgumentError("Last 2 dimensions of the array must be square"))
    size(a)[2] != size(b)[1] &&
        throw(
            ArgumentError(
                "Input operand 1 has a mismatch in its dimension 0, " *
                "with signature (m,m),(m,n)->(m,n) (size $(size(b)[1]) " *
                "is different from $(size(a)[2]))",
            ),
        )
    prod(size(a)) == 0 || prod(size(b)) == 0 && return zeros(T, size(b)...)
    x = zeros(T, size(b)...)
    solve_batched(a, b, x)
    return x
end

# Batched case: (...,m,m),(...,m,n)->(...,m,n)
function solve(a::NDArray{T,N}, b::NDArray{S,N}) where {T,S,N}
    size(a)[end - 1] != size(a)[end] &&
        throw(ArgumentError("Last 2 dimensions of the array must be square"))
    size(a)[end] != size(b)[end - 1] &&
        throw(
            ArgumentError(
                "Input operand 1 has a mismatch in its dimension " *
                "$(N-2), with signature (...,m,m),(...,m,n)->(...,m,n)" *
                " (size $(size(b)[end-1]) is different from $(size(a)[end]))",
            ),
        )
    prod(size(a)) == 0 || prod(size(b)) == 0 && return zeros(T, size(b)...)
    x = zeros(T, size(b)...)
    solve_batched(a, b, x)
    return x
end

# Mismatched batch dimensions
function solve(a::NDArray{T,N}, b::NDArray{S,M}) where {T,N,S,M}
    throw(ArgumentError("Batched matrices require signature (...,m,m),(...,m,n)->(...,m,n)"))
end

function svd(a::NDArray{T,2}, full_matrices::Bool=true) where {T}
    if size(a)[1] < size(a)[2]
        throw(ArgumentError("cuNumeric only supports M >= N"))
    end
    return _svd(a, full_matrices)
end

function svd(a::NDArray{T,1}, full_matrices::Bool=true) where {T}
    throw(ArgumentError("1-dimensional array given. Array must be at least two-dimensional"))
end

function svd(a::NDArray{T,N}, full_matrices::Bool=true) where {T,N}
    throw(ArgumentError("cuNumeric does not yet support stacked 2d arrays"))
end