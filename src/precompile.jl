# Compile representative host call paths without allocating arrays or starting
# a GPU/Legate runtime. The workload is inferred, never executed here.
let
    broadcast_workload(a, b, c) = a .+ b .+ c
    for T in (Float32, Float64)
        A = NDArray{T,1,Nothing}
        precompile(broadcast_workload, (A, A, A))
        precompile(rand, (Type{Complex{T}}, Int))
        precompile(randn, (Type{Complex{T}}, Int))
    end
end
