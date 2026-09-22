# Isolate layout and BLAS integer-interface effects using one allocation.
using CUDA, Statistics
const CB = CUDA.cuBLAS
function run(n)
    A = CUDA.ones(Float32, n, n)
    x = CUDA.ones(Float32, n)
    y = CUDA.zeros(Float32, n)
    println("cuBLAS version: ", CB.version()); flush(stdout)
    workspace_bytes = parse(Int, get(ENV, "GEMV_WORKSPACE_BYTES", "0"))
    workspace = CUDA.zeros(UInt8, workspace_bytes)
    workspace_bytes > 0 && CB.cublasSetWorkspace_v2(CB.handle(), workspace, workspace_bytes)
    get(ENV, "GEMV_TF32", "false") == "true" && CB.cublasSetMathMode(CB.handle(), CB.CUBLAS_TF32_TENSOR_OP_MATH)
    println("workspace_bytes=$workspace_bytes tf32=$(get(ENV, "GEMV_TF32", "false"))"); flush(stdout)
    GC.@preserve workspace begin
    for (bits, f) in ((32, CB.cublasSgemv_v2), (64, CB.cublasSgemv_v2_64)), trans in ('N', 'T')
        call() = f(CB.handle(), trans, n, n, 1f0, A, n, x, 1, 0f0, y, 1)
        for _ in 1:3
            call(); CUDA.synchronize()
        end
        samples = [(@elapsed begin call(); CUDA.synchronize() end)*1000 for _ in 1:5]
        @assert all(==(Float32(n)), Array(y))
        println("RESULT n=$n bits=$bits trans=$trans median_ms=$(median(samples)) samples=$samples"); flush(stdout)
    end
end
end
run(parse(Int, get(ARGS, 1, "65536")))
