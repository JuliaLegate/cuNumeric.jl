struct DaggerGrayScott{T,S,P}
    N::Int
    M::Int
    gpus::Int
    scope::S
    processors::P
    dt::T
    dx2::T
    cu::T
    cv::T
    f::T
    k::T
end

struct DaggerGrayScottState{A}
    U::A
    V::A
    Un::A
    Vn::A
end

function dagger_grayscott(::Type{T}, N, M, gpus, scope, processors) where {T}
    p = grayscott_gs_params(T)
    return DaggerGrayScott{T,typeof(scope),typeof(processors)}(
        N, M, gpus, scope, processors, p.dt, p.dx2, p.cu, p.cv, p.f, p.k
    )
end

function model_build_grayscott(config::ModelWorkerConfig)
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
    return dagger_grayscott(config.T, config.N, config.M, config.gpus, scope, processors)
end

function dagger_grayscott_state(b::DaggerGrayScott{T}, u_host, v_host) where {T}
    N, M = size(u_host)
    blocks = Dagger.Blocks(N, cld(M, b.gpus))
    assignment = reshape(copy(b.processors), 1, b.gpus)
    return Dagger.with_options(; scope=b.scope) do
        st = DaggerGrayScottState(
            Dagger.DArray(u_host, blocks, assignment), Dagger.DArray(v_host, blocks, assignment),
            Dagger.DArray(zeros(T, N, M), blocks, assignment),
            Dagger.DArray(zeros(T, N, M), blocks, assignment),
        )
        foreach(wait_for_darray, (st.U, st.V, st.Un, st.Vn))
        return st
    end
end

function model_initialize(b::DaggerGrayScott{T}) where {T}
    u_host, v_host = grayscott_host_init(T, b.N, b.M)
    return dagger_grayscott_state(b, u_host, v_host)
end

# @stencil handles cross-block halos; Wrap gives periodic BC. Double-buffered.
function model_run!(b::DaggerGrayScott, s::DaggerGrayScottState)
    dt, dx2, cu, cv, f, k = b.dt, b.dx2, b.cu, b.cv, b.f, b.k
    U, V, Un, Vn = s.U, s.V, s.Un, s.Vn
    Dagger.with_options(; scope=b.scope) do
        @stencil begin
            Un[idx] = begin
                nu = @neighbors(U[idx], 1, Wrap())
                up = nu[2, 2]
                vp = V[idx]
                lu = (nu[1, 2] + nu[3, 2] + nu[2, 1] + nu[2, 3] - 4up) / dx2
                up + dt * (cu * lu - up * vp * vp + f * (one(up) - up))
            end
            Vn[idx] = begin
                nv = @neighbors(V[idx], 1, Wrap())
                vp = nv[2, 2]
                up = U[idx]
                lv = (nv[1, 2] + nv[3, 2] + nv[2, 1] + nv[2, 3] - 4vp) / dx2
                vp + dt * (cv * lv + up * vp * vp - (f + k) * vp)
            end
            U[idx] = Un[idx]
            V[idx] = Vn[idx]
        end
    end
    return s.U
end

model_synchronize(::DaggerGrayScott) = Dagger.gpu_synchronize(:CUDA)
# Timesteps form one trajectory; do not fence between iterations.
model_fence_each_iteration(::DaggerGrayScott) = false

function model_correctness_context(b::DaggerGrayScott, config)
    n = min(32, b.N, b.M)
    return (; reference="CPU", dims=(n, n))
end

function model_check_correctness(b::DaggerGrayScott{T}, config) where {T}
    n = min(32, b.N, b.M)
    steps = config.n_correctness_iter
    u0, v0 = grayscott_host_init(T, n, n; deterministic=true)
    check = dagger_grayscott(T, n, n, b.gpus, b.scope, b.processors)
    s = dagger_grayscott_state(check, copy(u0), copy(v0))
    for _ in 1:steps
        model_run!(check, s)
    end
    model_synchronize(check)
    gu, gv = collect(s.U), collect(s.V)
    cuu, cvv = grayscott_cpu_steps(T, u0, v0, steps, grayscott_gs_params(T))
    return grayscott_correctness_status(gu, gv, cuu, cvv, T)
end
