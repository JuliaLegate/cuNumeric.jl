struct JACCGrayScott{T}
    N::Int
    M::Int
    gpus::Int
    dt::T
    dx2::T
    cu::T
    cv::T
    f::T
    k::T
end

function jacc_grayscott(::Type{T}, N, M, gpus) where {T}
    p = grayscott_gs_params(T)
    return JACCGrayScott{T}(N, M, gpus, p.dt, p.dx2, p.cu, p.cv, p.f, p.k)
end

mutable struct JACCGrayScottState{A}
    u::A
    v::A
    u_new::A
    v_new::A
end

function model_build_grayscott(config::ModelWorkerConfig)
    config.gpus == 1 || error(
        "JACC grayscott is single-GPU; multi-GPU is deferred pending JACC's 2D ghost fix"
    )
    return jacc_grayscott(config.T, config.N, config.M, config.gpus)
end

# Fully-periodic forward-Euler step; every cell wraps its four neighbors.
@inline function jacc_grayscott_kernel(i, j, u, v, u_new, v_new, N, M, dt, dx2, cu, cv, f, k)
    @inbounds begin
        up, vp = u[i, j], v[i, j]
        im = ifelse(i == 1, N, i - 1)
        ip = ifelse(i == N, 1, i + 1)
        jm = ifelse(j == 1, M, j - 1)
        jp = ifelse(j == M, 1, j + 1)
        lu = (u[ip, j] - 2up + u[im, j]) / dx2 + (u[i, jp] - 2up + u[i, jm]) / dx2
        lv = (v[ip, j] - 2vp + v[im, j]) / dx2 + (v[i, jp] - 2vp + v[i, jm]) / dx2
        uvv = up * vp * vp
        u_new[i, j] = up + dt * (cu * lu - uvv + f * (one(up) - up))
        v_new[i, j] = vp + dt * (cv * lv + uvv - (f + k) * vp)
    end
    return nothing
end

function jacc_grayscott_state(::Type{T}, u_host, v_host) where {T}
    N, M = size(u_host)
    return JACCGrayScottState(
        JACC.array(u_host), JACC.array(v_host),
        JACC.array(zeros(T, N, M)), JACC.array(zeros(T, N, M)),
    )
end

function model_initialize(b::JACCGrayScott{T}) where {T}
    JACC.Multi.ndev() >= 1 || error("JACC grayscott needs a visible GPU")
    u_host, v_host = grayscott_host_init(T, b.N, b.M)
    return jacc_grayscott_state(T, u_host, v_host)
end

function model_run!(b::JACCGrayScott, s::JACCGrayScottState)
    JACC.parallel_for(
        (b.N, b.M), jacc_grayscott_kernel,
        s.u, s.v, s.u_new, s.v_new, b.N, b.M, b.dt, b.dx2, b.cu, b.cv, b.f, b.k,
    )
    s.u, s.u_new = s.u_new, s.u
    s.v, s.v_new = s.v_new, s.v
    return s
end

model_synchronize(::JACCGrayScott) = JACC.synchronize()
# Timesteps form one trajectory; do not fence between iterations.
model_fence_each_iteration(::JACCGrayScott) = false

function model_correctness_context(b::JACCGrayScott, config)
    n = min(32, b.N, b.M)
    return (; reference="CPU", dims=(n, n))
end

function model_check_correctness(b::JACCGrayScott{T}, config) where {T}
    n = min(32, b.N, b.M)
    steps = config.n_correctness_iter
    u0, v0 = grayscott_host_init(T, n, n; deterministic=true)
    check = jacc_grayscott(T, n, n, 1)
    s = jacc_grayscott_state(T, copy(u0), copy(v0))
    for _ in 1:steps
        model_run!(check, s)
    end
    model_synchronize(check)
    gu, gv = JACC.to_host(s.u), JACC.to_host(s.v)
    cu, cv = grayscott_cpu_steps(T, u0, v0, steps, grayscott_gs_params(T))
    return grayscott_correctness_status(gu, gv, cu, cv, T)
end
