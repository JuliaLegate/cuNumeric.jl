# Infer representative host call paths without executing them: no arrays are
# allocated and no GPU/Legate runtime is started by these workloads.
let
    array_workload(a, b, c) = a .+ b .+ c
    scalar_workload(a, scalar) = a .+ scalar
    nested_workload(a, scalar) = sin.(a .^ 2) .+ scalar
    inplace_workload(dest, a, b, scalar) = (dest .= a .* scalar .+ b)
    custom_scalar(x, y) = x / (one(x) + abs(y))
    custom_workload(a, b) = custom_scalar.(a, b)

    for T in (Float32, Float64)
        # Fusion uses distinct linear, 2D, and 3D kernel constructors.
        for N in (1, 2, 3)
            A = NDArray{T,N,Nothing}
            precompile(array_workload, (A, A, A))
            precompile(scalar_workload, (A, T))
            precompile(nested_workload, (A, T))
            precompile(inplace_workload, (A, A, A, T))
            precompile(custom_workload, (A, A))
        end
        precompile(rand, (Type{Complex{T}}, Int))
        precompile(randn, (Type{Complex{T}}, Int))
    end
end
