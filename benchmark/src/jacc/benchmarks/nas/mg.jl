# LIMITATION: JACC does not provide distributed three-dimensional halo
# exchange, so this implementation is single-GPU. Every MG operator and the
# final norm are nevertheless JACC kernels/reductions; no CUDA.jl kernel is
# used directly.

include(joinpath(@__DIR__, "..", "..", "..", "nas", "mg.jl"))

struct JACCNASMG
    class::String
    N::Int
    M::Int
end

struct JACCNASMGState
    u
    r
    rhs
    c
    launch
    norm_reducer
end

function model_build_nas_mg(config::ModelWorkerConfig)
    config.T === Float64 || error("NAS MG requires Float64")
    config.gpus == 1 || error("JACC NAS MG currently supports one GPU")
    class = uppercase(string(get(config.kwargs, :class, "S")))
    p = nas_mg_parameters(class)
    (config.N, config.M) == (p.n, p.n) || error(
        "NAS MG class $class requires N=M=$(p.n)"
    )
    return JACCNASMG(class, config.N, config.M)
end

function model_initialize(b::JACCNASMG)
    p = nas_mg_parameters(b.class)
    sizes = nas_mg_level_sizes(p)
    u = [JACC.zeros(Float64, n, n, n) for n in sizes]
    r = [JACC.zeros(Float64, n, n, n) for n in sizes]
    return JACCNASMGState(
        u, r, JACC.Array(nas_mg_rhs(p)), nas_mg_smoother(b.class),
        JACC.launch_spec(; sync=false),
        JACC.reducer(; range=p.n^3, type=Float64, sync=false),
    )
end

@inline function jacc_mg_decode(index, n, offset)
    q = index - 1
    i = q % n + offset
    j = (q ÷ n) % n + offset
    k = q ÷ (n*n) + offset
    return i, j, k
end

function jacc_mg_zero(index, out)
    @inbounds out[index] = 0.0
end

function jacc_mg_comm_x(index, out, n)
    q = index - 1
    j = q % (n - 2) + 2
    k = q ÷ (n - 2) + 2
    @inbounds begin
        out[1, j, k] = out[n - 1, j, k]
        out[n, j, k] = out[2, j, k]
    end
end

function jacc_mg_comm_y(index, out, n)
    q = index - 1
    i = q % n + 1
    k = q ÷ n + 2
    @inbounds begin
        out[i, 1, k] = out[i, n - 1, k]
        out[i, n, k] = out[i, 2, k]
    end
end

function jacc_mg_comm_z(index, out, n)
    q = index - 1
    i, j = q % n + 1, q ÷ n + 1
    @inbounds begin
        out[i, j, 1] = out[i, j, n - 1]
        out[i, j, n] = out[i, j, 2]
    end
end

function jacc_mg_comm3!(s::JACCNASMGState, out)
    n = size(out, 1)
    JACC.parallel_for(s.launch, (n - 2)^2, jacc_mg_comm_x, out, n)
    JACC.parallel_for(s.launch, n*(n - 2), jacc_mg_comm_y, out, n)
    JACC.parallel_for(s.launch, n*n, jacc_mg_comm_z, out, n)
    return out
end

function jacc_mg_resid(index, r, u, v, n)
    i, j, k = jacc_mg_decode(index, n - 2, 2)
    @inbounds r[i, j, k] =
        v[i, j, k] - NAS_MG_A[1]*u[i, j, k] -
        NAS_MG_A[3] * (
            u[i, j - 1, k - 1] + u[i, j + 1, k - 1] + u[i, j - 1, k + 1] + u[i, j + 1, k + 1] +
            u[i - 1, j, k - 1] + u[i + 1, j, k - 1] + u[i - 1, j, k + 1] + u[i + 1, j, k + 1] +
            u[i - 1, j - 1, k] + u[i + 1, j - 1, k] + u[i - 1, j + 1, k] + u[i + 1, j + 1, k]
        ) -
        NAS_MG_A[4] * (
            u[i - 1, j - 1, k - 1] + u[i + 1, j - 1, k - 1] + u[i - 1, j + 1, k - 1] +
            u[i + 1, j + 1, k - 1] +
            u[i - 1, j - 1, k + 1] + u[i + 1, j - 1, k + 1] + u[i - 1, j + 1, k + 1] +
            u[i + 1, j + 1, k + 1]
        )
end

function jacc_mg_resid!(s, r, u, v)
    n = size(r, 1)
    JACC.parallel_for(s.launch, (n - 2)^3, jacc_mg_resid, r, u, v, n)
    return jacc_mg_comm3!(s, r)
end

function jacc_mg_psinv(index, u, r, n, c)
    i, j, k = jacc_mg_decode(index, n - 2, 2)
    @inbounds u[i, j, k] +=
        c[1]*r[i, j, k] +
        c[2] * (
            r[i - 1, j, k] + r[i + 1, j, k] + r[i, j - 1, k] +
            r[i, j + 1, k] + r[i, j, k - 1] + r[i, j, k + 1]
        ) +
        c[3] * (
            r[i, j - 1, k - 1] + r[i, j + 1, k - 1] + r[i, j - 1, k + 1] + r[i, j + 1, k + 1] +
            r[i - 1, j, k - 1] + r[i + 1, j, k - 1] + r[i - 1, j, k + 1] + r[i + 1, j, k + 1] +
            r[i - 1, j - 1, k] + r[i + 1, j - 1, k] + r[i - 1, j + 1, k] + r[i + 1, j + 1, k]
        )
end

function jacc_mg_psinv!(s, u, r)
    n = size(u, 1)
    JACC.parallel_for(s.launch, (n - 2)^3, jacc_mg_psinv, u, r, n, s.c)
    return jacc_mg_comm3!(s, u)
end

function jacc_mg_restrict(index, coarse, fine, nc)
    i, j, k = jacc_mg_decode(index, nc - 2, 2)
    fi, fj, fk = 2i - 1, 2j - 1, 2k - 1
    @inbounds coarse[i, j, k] =
        0.5*fine[fi, fj, fk] +
        0.25 * (
            fine[fi - 1, fj, fk] + fine[fi + 1, fj, fk] + fine[fi, fj - 1, fk] +
            fine[fi, fj + 1, fk] + fine[fi, fj, fk - 1] + fine[fi, fj, fk + 1]
        ) +
        0.125 * (
            fine[fi, fj - 1, fk - 1] + fine[fi, fj + 1, fk - 1] + fine[fi, fj - 1, fk + 1] +
            fine[fi, fj + 1, fk + 1] +
            fine[fi - 1, fj, fk - 1] + fine[fi + 1, fj, fk - 1] + fine[fi - 1, fj, fk + 1] +
            fine[fi + 1, fj, fk + 1] +
            fine[fi - 1, fj - 1, fk] + fine[fi + 1, fj - 1, fk] + fine[fi - 1, fj + 1, fk] +
            fine[fi + 1, fj + 1, fk]
        ) +
        0.0625 * (
            fine[fi - 1, fj - 1, fk - 1] + fine[fi + 1, fj - 1, fk - 1] +
            fine[fi - 1, fj + 1, fk - 1] + fine[fi + 1, fj + 1, fk - 1] +
            fine[fi - 1, fj - 1, fk + 1] + fine[fi + 1, fj - 1, fk + 1] +
            fine[fi - 1, fj + 1, fk + 1] + fine[fi + 1, fj + 1, fk + 1]
        )
end

function jacc_mg_restrict!(s, coarse, fine)
    nc = size(coarse, 1)
    JACC.parallel_for(s.launch, (nc - 2)^3, jacc_mg_restrict, coarse, fine, nc)
    return jacc_mg_comm3!(s, coarse)
end

@inline jacc_mg_lerp(a, b, weight) = muladd(weight, b - a, a)

function jacc_mg_interp(index, fine, coarse, nf)
    i, j, k = jacc_mg_decode(index, nf, 1)
    qi, qj, qk = i - 1, j - 1, k - 1
    i0, j0, k0 = qi ÷ 2 + 1, qj ÷ 2 + 1, qk ÷ 2 + 1
    i1, j1, k1 = i0 + (qi % 2), j0 + (qj % 2), k0 + (qk % 2)
    wi, wj, wk = 0.5*(qi % 2), 0.5*(qj % 2), 0.5*(qk % 2)
    @inbounds begin
        z00 = jacc_mg_lerp(coarse[i0, j0, k0], coarse[i1, j0, k0], wi)
        z10 = jacc_mg_lerp(coarse[i0, j1, k0], coarse[i1, j1, k0], wi)
        z01 = jacc_mg_lerp(coarse[i0, j0, k1], coarse[i1, j0, k1], wi)
        z11 = jacc_mg_lerp(coarse[i0, j1, k1], coarse[i1, j1, k1], wi)
        z0 = jacc_mg_lerp(z00, z10, wj)
        z1 = jacc_mg_lerp(z01, z11, wj)
        fine[i, j, k] += jacc_mg_lerp(z0, z1, wk)
    end
end

function jacc_mg_interp!(s, fine, coarse)
    JACC.parallel_for(s.launch, length(fine), jacc_mg_interp, fine, coarse, size(fine, 1))
    return fine
end

function jacc_mg_fill!(s, out)
    JACC.parallel_for(s.launch, length(out), jacc_mg_zero, out)
    return out
end

function jacc_mg_cycle!(s)
    finest = length(s.u)
    for level in finest:-1:2
        jacc_mg_restrict!(s, s.r[level - 1], s.r[level])
    end
    jacc_mg_fill!(s, s.u[1])
    jacc_mg_psinv!(s, s.u[1], s.r[1])
    for level in 2:(finest - 1)
        jacc_mg_fill!(s, s.u[level])
        jacc_mg_interp!(s, s.u[level], s.u[level - 1])
        jacc_mg_resid!(s, s.r[level], s.u[level], s.r[level])
        jacc_mg_psinv!(s, s.u[level], s.r[level])
    end
    jacc_mg_interp!(s, s.u[end], s.u[end - 1])
    jacc_mg_resid!(s, s.r[end], s.u[end], s.rhs)
    jacc_mg_psinv!(s, s.u[end], s.r[end])
    return nothing
end

function jacc_mg_norm_term(index, residual, n)
    i, j, k = jacc_mg_decode(index, n, 2)
    return @inbounds abs2(residual[i, j, k])
end

function model_run!(b::JACCNASMG, s::JACCNASMGState)
    p = nas_mg_parameters(b.class)
    foreach(out -> jacc_mg_fill!(s, out), s.u)
    jacc_mg_resid!(s, s.r[end], s.u[end], s.rhs)
    s.norm_reducer(jacc_mg_norm_term, s.r[end], p.n)
    for _ in 1:p.niter
        jacc_mg_cycle!(s)
        jacc_mg_resid!(s, s.r[end], s.u[end], s.rhs)
    end
    s.norm_reducer(jacc_mg_norm_term, s.r[end], p.n)
    return s.norm_reducer.workspace.ret
end

model_synchronize(::JACCNASMG) = JACC.synchronize()

function model_check_correctness(b::JACCNASMG, config)
    p = nas_mg_parameters(b.class)
    result = model_run!(b, model_initialize(b))
    model_synchronize(b)
    norm = sqrt(only(JACC.to_host(result))/Float64(p.n)^3)
    return nas_mg_verified(b.class, norm) ? "pass" : "fail"
end

function model_correctness_context(b::JACCNASMG, config)
    return (; reference="NPB-GPU", dims=(b.N, b.N, b.N))
end
