# Run each backend in a separate Julia process using the same environment.
using LinearAlgebra, Statistics, Krylov
BLAS.set_num_threads(1)
backend = ARGS[1]
const ElementType = get(ENV, "BENCH_ELTYPE", "Float64") == "Float32" ? Float32 : Float64
const RTOL = ElementType === Float32 ? Float32(1e-5) : 1e-8

if backend == "cuNumeric"
    @eval using cuNumeric
    @eval begin
        make_array(a) = NDArray(a)
        function make_array(a::Matrix{T}) where T
            layout = get(ENV, "BENCH_LAYOUT", "row")
            layout == "transposed_storage" && return permutedims(NDArray(permutedims(a)))
            layout == "column" || return NDArray(a)
            # Diagnostic: retain Julia's column-major storage through Legate.
            store = cuNumeric.Legate.attach_external_col_major(a)
            ptr = cuNumeric.nda_store_to_ndarray(store.handle)
            finalize(store.handle)
            return NDArray(ptr, T, Val(2), a)
        end
        solve!(w, A, b) = @allowautofetch Krylov.cg!(w, A, b; atol=zero(ElementType), rtol=RTOL, itmax=200)
        synchronize(w) = cuNumeric.issue_execution_fence(; block=true)
        host_array(a) = Array(a)
    end
    cuNumeric.allowscalar(false)
elseif backend in ("Dagger", "DaggerPatched")
    @eval using Dagger, CUDA
    CUDA.allowscalar(false)
    if backend == "DaggerPatched"
        # Benchmark-only workaround: stock Dagger currently calls CPU BLAS.gemv!
        # on CuArrays. Keep the unmodified failure reproducible as "Dagger".
        @eval function Dagger.matvecmul!(y::CUDA.CuArray, trans::Char, A::CUDA.CuArray,
                                       x::CUDA.CuArray, α, β)
            opA = trans == 'N' ? A : trans == 'T' ? transpose(A) : adjoint(A)
            return mul!(y, opA, x, α, β)
        end
    end
    @eval begin
        function make_array(a)
            result = Dagger.distribute(a, Dagger.Blocks(size(a)...))
            wait(result)
            for chunk in result.chunks
                @assert fetch(chunk; raw=true) isa Dagger.Chunk{<:CUDA.CuArray}
            end
            return result
        end
        solve!(w, A, b) = Krylov.cg!(w, A, b; atol=zero(ElementType), rtol=RTOL, itmax=200)
        function synchronize(w)
            foreach(wait, (w.x, w.r, w.p, w.Ap))
            Dagger.gpu_synchronize(:CUDA)
        end
        host_array(a) = collect(a)
    end
elseif backend == "CuArray"
    @eval using CUDA
    CUDA.allowscalar(false)
    @eval begin
        make_array(a) = CUDA.CuArray(a)
        solve!(w, A, b) = Krylov.cg!(w, A, b; atol=zero(ElementType), rtol=RTOL, itmax=200)
        synchronize(w) = CUDA.synchronize()
        host_array(a) = Array(a)
    end
else
    error("Unknown backend: $backend")
end

const PROFILE_DRIVER = backend == "cuNumeric" ? cuNumeric.CUDACore : CUDA.CUDACore

function benchmark(n)
    GC.gc() # Reclaim arrays from a preceding size before allocating the next.
    diagonal = ElementType.(range(2.0, 4.0; length=n))
    offdiagonal = fill(ElementType(-0.5), n-1)
    Ah = Matrix(SymTridiagonal(diagonal, offdiagonal))
    bh = ElementType[sin(i) + 1.0 for i in 1:n]
    A, b = make_array(Ah), make_array(bh)
    w = Krylov.CgWorkspace(A, b)
    println("Allocated $backend $ElementType n=$n; starting warm-up")
    flush(stdout)
    # Warm up compilation and execution separately from measured solves.
    for _ in 1:2
        solve!(w, A, b)
        synchronize(w)
    end
    println("Warm-up complete; starting five timed solves")
    flush(stdout)
    elapsed = Float64[]
    for sample in 1:5
        GC.gc()
        synchronize(w)
        # Optional Nsight capture of one warmed solve, including all backend
        # threads. Profiled timings should not replace the unprofiled results.
        capture = get(ENV, "BENCH_PROFILE", "false") == "true" && sample == 1 &&
                  n == parse(Int, get(ENV, "BENCH_PROFILE_N", string(n)))
        capture && PROFILE_DRIVER.cuProfilerStart()
        start = time_ns()
        solve!(w, A, b)
        synchronize(w)
        push!(elapsed, (time_ns() - start) / 1e6)
        capture && PROFILE_DRIVER.cuProfilerStop()
        @assert w.stats.solved
    end
    # Check the exact stored coefficients in Float64 without allocating a
    # second huge dense host matrix. The timed GPU operator is still dense.
    reference = SymTridiagonal(Float64.(diagonal), Float64.(offdiagonal))
    residual = norm(reference * Float64.(host_array(w.x)) - Float64.(bh)) / norm(Float64.(bh))
    @assert residual <= RTOL
    println("RESULT,$backend,$ElementType,$n,", w.stats.niter, ",", median(elapsed), ",", minimum(elapsed), ",", maximum(elapsed), ",", residual, ",", join(elapsed, ";"))
    flush(stdout)
end

println("backend=$backend eltype=$ElementType rtol=$RTOL Julia=$VERSION threads=$(Threads.nthreads()) Krylov=$(pkgversion(Krylov))")
println("CUBLAS_WORKSPACE_CONFIG=", get(ENV, "CUBLAS_WORKSPACE_CONFIG", "<default>"), " BENCH_LAYOUT=", get(ENV, "BENCH_LAYOUT", "row"))
flush(stdout)
sizes = length(ARGS) > 1 ? parse.(Int, ARGS[2:end]) : [256, 1024, 4096]
if backend in ("Dagger", "DaggerPatched")
    try
        Dagger.with_options(; scope=Dagger.scope(cuda_gpu=1)) do
            foreach(benchmark, sizes)
        end
    catch err
        # Dagger's task printer may stringify an entire input matrix before
        # displaying the error. Print just the root exception for large cases.
        root = Dagger.Sch.unwrap_nested_exception(err)
        println(stderr, "BENCHMARK FAILED: ", typeof(root))
        showerror(IOContext(stderr, :limit => true), root)
        println(stderr)
        exit(1)
    end
else
    foreach(benchmark, sizes)
end
