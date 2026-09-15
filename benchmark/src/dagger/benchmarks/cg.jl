using LinearAlgebra: Tridiagonal, norm

struct DaggerCG{T,S,P}
    N::Int
    gpus::Int
    check_every::Int
    max_iter::Int
    scope::S
    processors::P
end

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
    blocks = Dagger.Blocks(cld(N, b.gpus))
    assignment = reshape(copy(b.processors), b.gpus)
    return Dagger.with_options(; scope=b.scope) do
        st = DaggerCGState(
            Dagger.DArray(zeros(T, N), blocks, assignment),
            Dagger.DArray(zeros(T, N), blocks, assignment),
            Dagger.DArray(zeros(T, N), blocks, assignment),
            Dagger.DArray(zeros(T, N), blocks, assignment),
        )
        foreach(wait_for_darray, (st.x, st.r, st.p, st.Ap))
        return st
    end
end

function model_initialize(b::DaggerCG)
    return dagger_cg_state(b)
end

# Solve tridiag(1,4,1)*x = 1/2 from zero. @stencil applies the matvec with a zero
# pad so the fixed band matrix is honored at the domain boundaries. Reductions use
# `dims=1` so the scalars stay device-resident 1-element DArrays (which broadcast
# back over the vectors); the residual only reaches the host at `check_every`,
# matching cuNumeric and avoiding a per-iteration host sync.
function model_run!(b::DaggerCG{T}, s::DaggerCGState) where {T}
    x, r, p, Ap = s.x, s.r, s.p, s.Ap
    fmin = floatmin(T)
    # An explicit init keeps CUDA's device mapreduce off the _InitialValue path.
    ddot(a, c) = sum(a .* c; dims=1, init=zero(T))
    rho = Dagger.with_options(; scope=b.scope) do
        x .= zero(T)
        r .= T(0.5)
        p .= r
        return ddot(r, r)
    end
    target = (T==Float32 ? 1e-5 : 1e-8)^2 * b.N/4
    for k in 1:b.max_iter
        rho = Dagger.with_options(; scope=b.scope) do
            @stencil begin
                Ap[idx] = begin
                    np = @neighbors(p[idx], 1, Pad(zero(T)))
                    T(4) * np[2] + np[1] + np[3]
                end
            end
            alpha = rho ./ max.(ddot(p, Ap), fmin)
            x .= x .+ alpha .* p
            r .= r .- alpha .* Ap
            next = ddot(r, r)
            p .= r .+ (next ./ max.(rho, fmin)) .* p
            return next
        end
        if k % b.check_every == 0 || k == b.max_iter
            rr = only(collect(rho))
            isfinite(rr) || error("CG produced a nonfinite residual")
            (rr <= target || b.max_iter == 1) && return k
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
    n = min(b.N, 32)
    small = dagger_cg(T, n, b.gpus, b.check_every, b.max_iter, b.scope, b.processors)
    s = dagger_cg_state(small)
    model_run!(small, s)
    model_synchronize(small)
    x = collect(s.x)
    A = Tridiagonal(ones(T, n-1), fill(T(4), n), ones(T, n-1))
    err = b.max_iter==1 ? x .- T(n/(12n-4)) : A*x .- T(0.5)
    return norm(err) <= (T==Float32 ? 2e-5 : 2e-8)*sqrt(n)/2 ? "pass" : "fail"
end
