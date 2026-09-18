# Dagger partitions independent EP streams across its requested CUDA scope.
# Device chunks use high-level map! with the common scalar EP function; there
# is no hand-written CUDA kernel. The untimed correctness pass uses Dagger's
# mapped reductions so only the two verification scalars reach the host.

include(joinpath(@__DIR__, "..", "..", "..", "nas", "ep.jl"))

struct DaggerNASEP{S,P}
    class::String
    N::Int
    batches::Int
    gpus::Int
    scope::S
    processors::P
end

struct DaggerNASEPState{P,I,PC,IC,S}
    partials::P
    indices::I
    partial_chunks::PC
    index_chunks::IC
    scopes::S
end

function model_build_nas_ep(config::ModelWorkerConfig)
    config.T === Float64 || error("NAS EP requires Float64")
    config.M == 1 || error("NAS EP requires M=1")
    class = uppercase(string(get(config.kwargs, :class, "S")))
    p = nas_ep_parameters(class)
    expected = nas_ep_random_numbers(p)
    config.N == expected || error("NAS EP class $class requires N=$expected")
    available = length(collect(CUDA.devices()))
    available == config.gpus || error(
        "Dagger sees $available GPU(s), but this run requested $(config.gpus)"
    )
    scope = Dagger.scope(; cuda_gpus=collect(1:config.gpus))
    processors = sort!(collect(Dagger.compatible_processors(scope)); by=string)
    length(processors) == config.gpus || error("Dagger CUDA processor count mismatch")
    return DaggerNASEP(
        class, config.N, nas_ep_batches(p), config.gpus, scope, processors
    )
end

function model_initialize(b::DaggerNASEP)
    block = cld(b.batches, b.gpus)
    return Dagger.with_options(; scope=b.scope) do
        partials = Dagger.DArray(
            fill(NASEPPartial(), b.batches), Dagger.Blocks(block), b.processors
        )
        indices = Dagger.DArray(
            collect(Int64, 0:(b.batches - 1)), Dagger.Blocks(block), b.processors
        )
        foreach(wait_for_darray, (partials, indices))
        partial_chunks = map(task -> fetch(task; raw=true), partials.chunks)
        index_chunks = map(task -> fetch(task; raw=true), indices.chunks)
        length(partial_chunks) == b.gpus || error("Dagger EP chunk count mismatch")
        processors = Dagger.processor.(partial_chunks)
        length(unique(processors)) == b.gpus || error(
            "Dagger did not place one EP chunk on each requested GPU"
        )
        return DaggerNASEPState(
            partials, indices, partial_chunks, index_chunks,
            Dagger.ExactScope.(processors),
        )
    end
end

function dagger_nas_ep_chunk!(partials, indices, jump)
    map!(i -> nas_ep_batch(i, jump), partials, indices)
    return nothing
end

dagger_nas_ep_sx(partial) = partial.sx
dagger_nas_ep_sy(partial) = partial.sy

function dagger_nas_ep_chunk_sum(f, partials)
    return mapreduce(f, +, partials; init=0.0)
end

function dagger_nas_ep_sum(f, s::DaggerNASEPState)
    partials = map(zip(s.partial_chunks, s.scopes)) do (chunk, scope)
        Dagger.@spawn scope=scope dagger_nas_ep_chunk_sum(f, chunk)
    end
    return mapreduce(fetch, +, partials; init=0.0)
end

function model_run!(b::DaggerNASEP, s::DaggerNASEPState)
    tasks = map(zip(s.partial_chunks, s.index_chunks, s.scopes)) do (out, indices, scope)
        Dagger.@spawn scope=scope dagger_nas_ep_chunk!(out, indices, nas_ep_batch_jump())
    end
    foreach(fetch, tasks)
    return s.partials
end

model_synchronize(::DaggerNASEP) = Dagger.gpu_synchronize(:CUDA)
model_throughput_label(::DaggerNASEP) = "G random numbers/s"

function model_check_correctness(b::DaggerNASEP, config)
    state = model_initialize(b)
    model_run!(b, state)
    model_synchronize(b)
    sx = dagger_nas_ep_sum(dagger_nas_ep_sx, state)
    sy = dagger_nas_ep_sum(dagger_nas_ep_sy, state)
    return nas_ep_verified(b.class, sx, sy) ? "pass" : "fail"
end

function model_correctness_context(b::DaggerNASEP, config)
    return (; reference="NPB-GPU", dims=(b.N, 1))
end
