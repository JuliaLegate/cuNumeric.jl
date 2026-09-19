# LIMITATION: Dagger has no native multigrid transfer operator. Residuals and
# smoothers use Dagger's distributed periodic stencils; restriction and
# interpolation use Dagger-scheduled chunk broadcasts. This avoids scalar
# DArray indexing and does not use a hand-written CUDA kernel, but interpolation
# currently builds seven temporary DArrays per level.
# Restriction filters the entire fine grid before subsampling (eight times as
# many stencil outputs as direct coarse-grid evaluation). Coarse levels use
# fewer GPUs; stencil/chunk movement is runtime-managed, not explicit minimal
# halo exchange. Norms produce device-local partials; global aggregation is
# untimed, unlike other adapters. Common timing deviations: see nas/README.md.

include(joinpath(@__DIR__, "..", "..", "..", "nas", "mg.jl"))

import Dagger: @stencil, Wrap

struct DaggerNASMG{S,P}
    class::String
    N::Int
    M::Int
    gpus::Int
    scope::S
    processors::P
end

struct DaggerNASMGState{U,R,V}
    u::U
    r::R
    rhs::V
end

function model_build_nas_mg(config::ModelWorkerConfig)
    config.T === Float64 || error("NAS MG requires Float64")
    class = uppercase(string(get(config.kwargs, :class, "S")))
    p = nas_mg_parameters(class)
    (config.N, config.M) == (p.n, p.n) || error(
        "NAS MG class $class requires N=M=$(p.n)"
    )
    available = length(collect(CUDA.devices()))
    available == config.gpus || error(
        "Dagger sees $available GPU(s), but this run requested $(config.gpus)"
    )
    scope = Dagger.scope(; cuda_gpus=collect(1:config.gpus))
    processors = sort!(collect(Dagger.compatible_processors(scope)); by=string)
    length(processors) == config.gpus || error("Dagger CUDA processor count mismatch")
    return DaggerNASMG(class, config.N, config.M, config.gpus, scope, processors)
end

dagger_mg_level_sizes(p) = [2^level for level in 1:round(Int, log2(p.n))]

function dagger_nas_mg_array(host, b::DaggerNASMG)
    n = size(host, 1)
    block = cld(n, b.gpus)
    nchunks = cld(n, block)
    assignment = reshape(copy(b.processors[1:nchunks]), 1, 1, nchunks)
    return Dagger.DArray(host, Dagger.Blocks(n, n, block), assignment)
end

function model_initialize(b::DaggerNASMG)
    p = nas_mg_parameters(b.class)
    return Dagger.with_options(; scope=b.scope) do
        sizes = dagger_mg_level_sizes(p)
        u = [dagger_nas_mg_array(zeros(Float64, n, n, n), b) for n in sizes]
        r = [dagger_nas_mg_array(zeros(Float64, n, n, n), b) for n in sizes]
        ghosted_rhs = nas_mg_rhs(p)
        rhs_host = copy(@view ghosted_rhs[2:(end - 1), 2:(end - 1), 2:(end - 1)])
        rhs = dagger_nas_mg_array(rhs_host, b)
        foreach(wait_for_darray, u)
        foreach(wait_for_darray, r)
        wait_for_darray(rhs)
        return DaggerNASMGState(u, r, rhs)
    end
end

@inline function dagger_mg_resid_value(v, n)
    edges =
        n[2, 1, 1] + n[2, 3, 1] + n[2, 1, 3] + n[2, 3, 3] +
        n[1, 2, 1] + n[3, 2, 1] + n[1, 2, 3] + n[3, 2, 3] +
        n[1, 1, 2] + n[3, 1, 2] + n[1, 3, 2] + n[3, 3, 2]
    corners =
        n[1, 1, 1] + n[3, 1, 1] + n[1, 3, 1] + n[3, 3, 1] +
        n[1, 1, 3] + n[3, 1, 3] + n[1, 3, 3] + n[3, 3, 3]
    return v - NAS_MG_A[1]*n[2, 2, 2] - NAS_MG_A[3]*edges - NAS_MG_A[4]*corners
end

function dagger_mg_resid!(r, u, v)
    @stencil r[idx] = dagger_mg_resid_value(v[idx], @neighbors(u[idx], 1, Wrap()))
    return r
end

@inline function dagger_mg_psinv_value(n, c)
    faces = n[1, 2, 2] + n[3, 2, 2] + n[2, 1, 2] + n[2, 3, 2] +
            n[2, 2, 1] + n[2, 2, 3]
    edges =
        n[2, 1, 1] + n[2, 3, 1] + n[2, 1, 3] + n[2, 3, 3] +
        n[1, 2, 1] + n[3, 2, 1] + n[1, 2, 3] + n[3, 2, 3] +
        n[1, 1, 2] + n[3, 1, 2] + n[1, 3, 2] + n[3, 3, 2]
    return c[1]*n[2, 2, 2] + c[2]*faces + c[3]*edges
end

function dagger_mg_psinv!(u, r, c)
    @stencil u[idx] = u[idx] + dagger_mg_psinv_value(@neighbors(r[idx], 1, Wrap()), c)
    return u
end

@inline function dagger_mg_restrict_value(n)
    faces = n[1, 2, 2] + n[3, 2, 2] + n[2, 1, 2] + n[2, 3, 2] +
            n[2, 2, 1] + n[2, 2, 3]
    edges =
        n[2, 1, 1] + n[2, 3, 1] + n[2, 1, 3] + n[2, 3, 3] +
        n[1, 2, 1] + n[3, 2, 1] + n[1, 2, 3] + n[3, 2, 3] +
        n[1, 1, 2] + n[3, 1, 2] + n[1, 3, 2] + n[3, 3, 2]
    corners =
        n[1, 1, 1] + n[3, 1, 1] + n[1, 3, 1] + n[3, 3, 1] +
        n[1, 1, 3] + n[3, 1, 3] + n[1, 3, 3] + n[3, 3, 3]
    return 0.5*n[2, 2, 2] + 0.25*faces + 0.125*edges + 0.0625*corners
end

function dagger_mg_covering_chunk(array, first_z, last_z)
    for index in CartesianIndices(array.chunks)
        zrange = array.subdomains[index].indexes[3]
        first_z >= first(zrange) && last_z <= last(zrange) && return index, zrange
    end
    return error("MG transfer range $first_z:$last_z crosses Dagger chunks")
end

function dagger_mg_downsample_chunk!(coarse, fine, first_local_z)
    stop_local_z = first_local_z + 2*(size(coarse, 3) - 1)
    coarse .= @view fine[
        2:2:size(fine, 1), 2:2:size(fine, 2), first_local_z:2:stop_local_z
    ]
    return nothing
end

function dagger_mg_downsample!(coarse, filtered)
    Dagger.spawn_datadeps() do
        for index in CartesianIndices(coarse.chunks)
            target_z = coarse.subdomains[index].indexes[3]
            first_z, last_z = 2first(target_z), 2last(target_z)
            source_index, source_z = dagger_mg_covering_chunk(filtered, first_z, last_z)
            local_z = first_z - first(source_z) + 1
            Dagger.@spawn dagger_mg_downsample_chunk!(
                Dagger.Out(coarse.chunks[index]),
                Dagger.In(filtered.chunks[source_index]),
                local_z,
            )
        end
    end
    return coarse
end

function dagger_mg_restrict!(coarse, fine)
    filtered = @stencil dagger_mg_restrict_value(@neighbors(fine[idx], 1, Wrap()))
    return dagger_mg_downsample!(coarse, filtered)
end

@inline dagger_mg_interp_100(n) = 0.5*(n[1, 2, 2] + n[2, 2, 2])
@inline dagger_mg_interp_010(n) = 0.5*(n[2, 1, 2] + n[2, 2, 2])
@inline dagger_mg_interp_001(n) = 0.5*(n[2, 2, 1] + n[2, 2, 2])
@inline dagger_mg_interp_110(n) =
    0.25*(n[1, 1, 2] + n[2, 1, 2] + n[1, 2, 2] + n[2, 2, 2])
@inline dagger_mg_interp_101(n) =
    0.25*(n[1, 2, 1] + n[2, 2, 1] + n[1, 2, 2] + n[2, 2, 2])
@inline dagger_mg_interp_011(n) =
    0.25*(n[2, 1, 1] + n[2, 2, 1] + n[2, 1, 2] + n[2, 2, 2])
@inline dagger_mg_interp_111(n) =
    0.125*(
        n[1, 1, 1] + n[2, 1, 1] + n[1, 2, 1] + n[2, 2, 1] +
        n[1, 1, 2] + n[2, 1, 2] + n[1, 2, 2] + n[2, 2, 2]
    )

function dagger_mg_interp_components(coarse)
    c100 = @stencil dagger_mg_interp_100(@neighbors(coarse[idx], 1, Wrap()))
    c010 = @stencil dagger_mg_interp_010(@neighbors(coarse[idx], 1, Wrap()))
    c110 = @stencil dagger_mg_interp_110(@neighbors(coarse[idx], 1, Wrap()))
    c001 = @stencil dagger_mg_interp_001(@neighbors(coarse[idx], 1, Wrap()))
    c101 = @stencil dagger_mg_interp_101(@neighbors(coarse[idx], 1, Wrap()))
    c011 = @stencil dagger_mg_interp_011(@neighbors(coarse[idx], 1, Wrap()))
    c111 = @stencil dagger_mg_interp_111(@neighbors(coarse[idx], 1, Wrap()))
    return (coarse, c100, c010, c110, c001, c101, c011, c111)
end

function dagger_mg_upsample_chunk!(fine, global_z, coarse_z, components...)
    nx, ny, nz = size(fine)
    for odd_z in (false, true)
        local_z = isodd(global_z) == odd_z ? 1 : 2
        local_z > nz && continue
        output_z = local_z:2:nz
        first_global_z = global_z + local_z - 1
        first_coarse_z = cld(first_global_z, 2) - coarse_z + 1
        input_z = first_coarse_z:(first_coarse_z + length(output_z) - 1)
        zbit = odd_z ? 4 : 0
        for odd_y in (false, true), odd_x in (false, true)
            output_x = odd_x ? (1:2:nx) : (2:2:nx)
            output_y = odd_y ? (1:2:ny) : (2:2:ny)
            component = components[1 + (odd_x ? 1 : 0) + (odd_y ? 2 : 0) + zbit]
            @views fine[output_x, output_y, output_z] .+= component[:, :, input_z]
        end
    end
    return nothing
end

function dagger_mg_upsample!(fine, components)
    coarse = first(components)
    Dagger.spawn_datadeps() do
        for index in CartesianIndices(fine.chunks)
            target_z = fine.subdomains[index].indexes[3]
            first_coarse_z = cld(first(target_z), 2)
            last_coarse_z = cld(last(target_z), 2)
            source_index, source_z = dagger_mg_covering_chunk(
                coarse, first_coarse_z, last_coarse_z
            )
            chunks = map(component -> component.chunks[source_index], components)
            Dagger.@spawn dagger_mg_upsample_chunk!(
                Dagger.Out(fine.chunks[index]),
                first(target_z),
                first(source_z),
                Dagger.In(chunks[1]),
                Dagger.In(chunks[2]),
                Dagger.In(chunks[3]),
                Dagger.In(chunks[4]),
                Dagger.In(chunks[5]),
                Dagger.In(chunks[6]),
                Dagger.In(chunks[7]),
                Dagger.In(chunks[8]),
            )
        end
    end
    return fine
end

dagger_mg_interp!(fine, coarse) =
    dagger_mg_upsample!(fine, dagger_mg_interp_components(coarse))

function dagger_mg_cycle!(s, c)
    finest = length(s.u)
    for level in finest:-1:2
        dagger_mg_restrict!(s.r[level - 1], s.r[level])
    end
    fill!(s.u[1], 0.0)
    dagger_mg_psinv!(s.u[1], s.r[1], c)
    for level in 2:(finest - 1)
        fill!(s.u[level], 0.0)
        dagger_mg_interp!(s.u[level], s.u[level - 1])
        dagger_mg_resid!(s.r[level], s.u[level], s.r[level])
        dagger_mg_psinv!(s.u[level], s.r[level], c)
    end
    dagger_mg_interp!(s.u[end], s.u[end - 1])
    dagger_mg_resid!(s.r[end], s.u[end], s.rhs)
    dagger_mg_psinv!(s.u[end], s.r[end], c)
    return nothing
end

function dagger_mg_norm_chunk(values)
    dims = ntuple(identity, ndims(values))
    return mapreduce(abs2, +, values; dims, init=zero(eltype(values)))
end

function dagger_mg_norm_tasks(b::DaggerNASMG, residual)
    length(residual.chunks) == length(b.processors) || error(
        "Dagger MG expected one finest-level slab per GPU"
    )
    tasks = Vector{Dagger.DTask}(undef, length(b.processors))
    Dagger.spawn_datadeps() do
        for i in eachindex(tasks)
            tasks[i] = Dagger.@spawn scope=Dagger.ExactScope(b.processors[i]) dagger_mg_norm_chunk(
                Dagger.In(residual.chunks[i])
            )
        end
    end
    return tasks
end

function model_run!(b::DaggerNASMG, s::DaggerNASMGState)
    p = nas_mg_parameters(b.class)
    return Dagger.with_options(; scope=b.scope) do
        foreach(x -> fill!(x, 0.0), s.u)
        dagger_mg_resid!(s.r[end], s.u[end], s.rhs)
        dagger_mg_norm_tasks(b, s.r[end])
        c = nas_mg_smoother(b.class)
        for _ in 1:p.niter
            dagger_mg_cycle!(s, c)
            dagger_mg_resid!(s.r[end], s.u[end], s.rhs)
        end
        wait_for_darray(s.r[end])
        return dagger_mg_norm_tasks(b, s.r[end])
    end
end

model_synchronize(::DaggerNASMG) = Dagger.gpu_synchronize(:CUDA)

function model_check_correctness(b::DaggerNASMG, config)
    p = nas_mg_parameters(b.class)
    tasks = model_run!(b, model_initialize(b))
    model_synchronize(b)
    squared = sum(only(fetch(task)) for task in tasks)
    norm = sqrt(squared/Float64(p.n)^3)
    return nas_mg_verified(b.class, norm) ? "pass" : "fail"
end

function model_correctness_context(b::DaggerNASMG, config)
    return (; reference="NPB-GPU", dims=(b.N, b.N, b.N))
end
