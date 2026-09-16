using LinearAlgebra: Tridiagonal, norm

struct DaggerCG{T,S,P}
    N::Int
    gpus::Int
    check_every::Int
    max_iter::Int
    scope::S
    processors::P
end

# Mutable GPU buffers held on the Dagger processor; datadeps sequences the tasks
# that mutate them in place.
struct DaggerCGState{X,R}
    x::X
    r::X
    p::X
    Ap::X
    rho::R
end

function dagger_cg(::Type{T}, N, gpus, check_every, max_iter, scope, processors) where {T}
    return DaggerCG{T,typeof(scope),typeof(processors)}(
        N, gpus, check_every, max_iter, scope, processors
    )
end

function model_build_cg(config::ModelWorkerConfig)
    config.M == 1 || error("Dagger CG requires M=1")
    config.gpus == 1 || error("Dagger CG is single-GPU for now")
    available = length(collect(CUDA.devices()))
    available == config.gpus || error(
        "Dagger sees $available GPU(s), but this run was planned for $(config.gpus). " *
        "Set CUDA_VISIBLE_DEVICES to exactly the selected devices.",
    )
    scope = Dagger.scope(; cuda_gpus=collect(1:config.gpus))
    processors = sort!(collect(Dagger.compatible_processors(scope)); by=string)
    length(processors) == config.gpus || error(
        "Dagger CUDA scope contains $(length(processors)) processor(s), expected $(config.gpus)"
    )
    check_every = Int(get(config.kwargs, :check_every, 10))
    max_iter = Int(get(config.kwargs, :max_iter, 1000))
    return dagger_cg(config.T, config.N, config.gpus, check_every, max_iter, scope, processors)
end

function dagger_cg_state(b::DaggerCG{T}) where {T}
    N = b.N
    # Bind to the GPU processor (not just the scope) so the mutable chunk is
    # tagged on-device; a scope-only @mutable leaves proc=OSProc and datadeps
    # then tries to move the CuArray to the host.
    proc = first(b.processors)
    x = Dagger.@mutable processor = proc CUDA.zeros(T, N)
    r = Dagger.@mutable processor = proc CUDA.zeros(T, N)
    p = Dagger.@mutable processor = proc CUDA.zeros(T, N)
    Ap = Dagger.@mutable processor = proc CUDA.zeros(T, N)
    rho = Dagger.@mutable processor = proc CUDA.zeros(T, 1)
    return DaggerCGState(x, r, p, Ap, rho)
end

model_initialize(b::DaggerCG) = dagger_cg_state(b)

# One coarse task per iteration: the whole tridiag(1,4,1) CG step runs on the GPU
# buffer, keeping alpha/beta/residual as device 1-element arrays. Reductions use
# an explicit init so CUDA's device mapreduce stays off the _InitialValue path.
function cg_reset!(x, r, p, rho, ::Type{T}) where {T}
    x .= zero(T)
    r .= T(0.5)
    p .= r
    rho .= sum(r .* r; dims=1, init=zero(T))
    return nothing
end

function cg_step_chunk!(x, r, p, Ap, rho, fmin, ::Type{T}) where {T}
    N = length(p)
    @views Ap .= T(4) .* p
    @views Ap[2:N] .+= p[1:(N - 1)]
    @views Ap[1:(N - 1)] .+= p[2:N]
    alpha = rho ./ max.(sum(p .* Ap; dims=1, init=zero(T)), fmin)
    x .+= alpha .* p
    r .-= alpha .* Ap
    next = sum(r .* r; dims=1, init=zero(T))
    p .= r .+ (next ./ max.(rho, fmin)) .* p
    rho .= next
    return nothing
end

# Solve tridiag(1,4,1)*x = 1/2 from zero. Each datadeps region runs `check_every`
# iterations as a serial chain, then the residual reaches the host once per check.
function model_run!(b::DaggerCG{T}, s::DaggerCGState) where {T}
    fmin = floatmin(T)
    x, r, p, Ap, rho = s.x, s.r, s.p, s.Ap, s.rho
    Dagger.with_options(; scope=b.scope) do
        Dagger.spawn_datadeps() do
            Dagger.@spawn cg_reset!(Dagger.Out(x), Dagger.Out(r), Dagger.Out(p), Dagger.Out(rho), T)
        end
    end
    target = (T==Float32 ? 1e-5 : 1e-8)^2 * b.N/4
    k = 0
    while k < b.max_iter
        block = min(b.check_every, b.max_iter - k)
        Dagger.with_options(; scope=b.scope) do
            Dagger.spawn_datadeps() do
                for _ in 1:block
                    Dagger.@spawn cg_step_chunk!(
                        Dagger.InOut(x), Dagger.InOut(r), Dagger.InOut(p), Dagger.InOut(Ap),
                        Dagger.InOut(rho), fmin, T,
                    )
                end
            end
        end
        k += block
        rr = only(Array(fetch(rho)))
        isfinite(rr) || error("CG produced a nonfinite residual")
        (rr <= target || b.max_iter == 1) && return k
    end
    return error("CG did not converge within max_iter")
end

model_synchronize(::DaggerCG) = Dagger.gpu_synchronize(:CUDA)

function model_correctness_context(b::DaggerCG, config)
    n = min(32, b.N)
    return (; reference="CPU", dims=(n, 1))
end

function model_check_correctness(b::DaggerCG{T}, config) where {T}
    n = min(b.N, 32)
    small = dagger_cg(T, n, b.gpus, b.check_every, b.max_iter, b.scope, b.processors)
    s = dagger_cg_state(small)
    model_run!(small, s)
    model_synchronize(small)
    x = Array(fetch(s.x))
    A = Tridiagonal(ones(T, n-1), fill(T(4), n), ones(T, n-1))
    err = b.max_iter==1 ? x .- T(n/(12n-4)) : A*x .- T(0.5)
    return norm(err) <= (T==Float32 ? 2e-5 : 2e-8)*sqrt(n)/2 ? "pass" : "fail"
end
