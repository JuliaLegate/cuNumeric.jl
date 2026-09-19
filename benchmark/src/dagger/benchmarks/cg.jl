using LinearAlgebra: Tridiagonal, norm

struct DaggerCG{T,S,P}
    N::Int
    gpus::Int
    check_every::Int
    max_iter::Int
    scope::S
    processors::P
end

# Distributed vectors share one partitioning across the selected GPUs.
struct DaggerCGState{A}
    x::A
    r::A
    p::A
    Ap::A
end

function dagger_cg(::Type{T}, N, gpus, check_every, max_iter, scope, processors) where {T}
    return DaggerCG{T,typeof(scope),typeof(processors)}(
        N, gpus, check_every, max_iter, scope, processors
    )
end

function model_build_cg(config::ModelWorkerConfig)
    config.M == 1 || error("Dagger CG requires M=1")
    config.N % config.gpus == 0 || error("Dagger CG requires N divisible by GPUs")
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
    blocks = Dagger.Blocks(b.N ÷ b.gpus)
    assignment = copy(b.processors)
    return Dagger.with_options(; scope=b.scope) do
        st = DaggerCGState(
            zeros(blocks, T, b.N; assignment), zeros(blocks, T, b.N; assignment),
            zeros(blocks, T, b.N; assignment), zeros(blocks, T, b.N; assignment),
        )
        foreach(wait_for_darray, (st.x, st.r, st.p, st.Ap))
        return st
    end
end

model_initialize(b::DaggerCG) = dagger_cg_state(b)

function dagger_cg_chunk_dot(x, y, ::Type{T}) where {T}
    return mapreduce(*, +, x, y; init=zero(T))
end

function dagger_cg_dot(b::DaggerCG{T}, x, y) where {T}
    partials = map(eachindex(b.processors)) do i
        Dagger.@spawn scope=Dagger.ExactScope(b.processors[i]) dagger_cg_chunk_dot(
            x.chunks[i], y.chunks[i], T
        )
    end
    return mapreduce(fetch, +, partials; init=zero(T))
end

# Idiomatic distributed-array CG. The tridiagonal multiply uses Dagger's halo
# handling; vector updates use broadcasts and dot products reduce each chunk.
function model_run!(b::DaggerCG{T}, s::DaggerCGState) where {T}
    x, r, p, Ap = s.x, s.r, s.p, s.Ap
    Dagger.with_options(; scope=b.scope) do
        x .= zero(T)
        r .= T(0.5)
        p .= r
        rho = dagger_cg_dot(b, r, r)
        target = (T==Float32 ? 1e-5 : 1e-8)^2 * b.N/4
        for k in 1:b.max_iter
            @stencil Ap[idx] = begin
                np = @neighbors(p[idx], 1, Pad(zero(T)))
                np[1] + T(4)*np[2] + np[3]
            end
            alpha = rho/max(dagger_cg_dot(b, p, Ap), floatmin(T))
            x .+= alpha .* p
            r .-= alpha .* Ap
            next = dagger_cg_dot(b, r, r)
            p .= r .+ (next/max(rho, floatmin(T))) .* p
            rho = next
            if k % b.check_every == 0 || k == b.max_iter
                isfinite(rho) || error("CG produced a nonfinite residual")
                (rho <= target || b.max_iter == 1) && return k
            end
        end
    end
    return error("CG did not converge within max_iter")
end

model_synchronize(::DaggerCG) = Dagger.gpu_synchronize(:CUDA)

function model_correctness_context(b::DaggerCG, config)
    n = min(32, b.N)
    return (; reference="CPU", dims=(n, 1))
end

function model_check_correctness(b::DaggerCG{T}, config) where {T}
    n = min(b.N, max(b.gpus, fld(32, b.gpus) * b.gpus))
    small = dagger_cg(T, n, b.gpus, b.check_every, b.max_iter, b.scope, b.processors)
    s = dagger_cg_state(small)
    model_run!(small, s)
    model_synchronize(small)
    x = collect(s.x)
    A = Tridiagonal(ones(T, n-1), fill(T(4), n), ones(T, n-1))
    err = b.max_iter==1 ? x .- T(n/(12n-4)) : A*x .- T(0.5)
    return norm(err) <= (T==Float32 ? 2e-5 : 2e-8)*sqrt(n)/2 ? "pass" : "fail"
end
