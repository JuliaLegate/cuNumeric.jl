# Infer representative host call paths without executing them: no arrays are
# allocated and no GPU/Legate runtime is started by these workloads.
let
    array_workload(a, b, c) = a .+ b .+ c
    scalar_workload(a, scalar) = a .+ scalar
    nested_workload(a, scalar) = sin.(a .^ 2) .+ scalar
    inplace_workload(dest, a, b, scalar) = (dest .= a .* scalar .+ b)
    custom_scalar(x, y) = x / (one(x) + abs(y))
    custom_workload(a, b) = custom_scalar.(a, b)
    full_mapreduce_workload(a) = mapreduce(abs2, +, a)
    dims_mapreduce_workload(a) = mapreduce(abs2, +, a; dims=1)

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
        precompile(full_mapreduce_workload, (NDArray{T,1,Nothing},))
        precompile(dims_mapreduce_workload, (NDArray{T,2,Nothing},))
        precompile(rand, (Type{Complex{T}}, Int))
        precompile(randn, (Type{Complex{T}}, Int))
    end
end

# Macro expansion is pure syntax work. Exercise each scope form without
# evaluating the generated code or starting Legate/CUDA. Running the passes
# captures runtime-dispatched Expr/Any callees that a precompile signature alone
# cannot reach. The workload does not depend on array element types or ranks.
using PrecompileTools: @compile_workload

@compile_workload begin
    for input in (
        :(@accelerate function update!(dest, a, b; scale=2f0)
            product = @. a * b
            dest .+= scale .* product
            return dest
        end),
        :(@accelerate let
            shifted = a[2:end] .+ b[1:(end - 1)]
            sin.(shifted .^ 2)
        end),
        :(@accelerate begin
            product = a[2:end] .* b[2:end]
            shifted = product .+ 1f0
            (product, shifted)
        end),
        :(@accelerate a .+ b .* 2f0),
    )
        Base.macroexpand(@__MODULE__, input)
    end
end
