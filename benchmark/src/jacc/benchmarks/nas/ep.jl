# JACC.Multi partitions the independent EP streams across all visible GPUs.
# Each stream executes the exact NPB 46-bit RNG and writes one partial result;
# host aggregation is performed only by the untimed correctness check.
# LIMITATION: Like the other EP adapters, timing excludes global aggregation.
# JACC.Multi synchronizes launches; this is not an asynchronous multi-GPU DAG.

include(joinpath(@__DIR__, "..", "..", "..", "nas", "ep.jl"))

struct JACCNASEP
    class::String
    N::Int
    batches::Int
    gpus::Int
end

struct JACCNASEPState{Q,V}
    q::Q
    sx::V
    sy::V
end

function model_build_nas_ep(config::ModelWorkerConfig)
    config.T === Float64 || error("NAS EP requires Float64")
    config.M == 1 || error("NAS EP requires M=1")
    class = uppercase(string(get(config.kwargs, :class, "S")))
    p = nas_ep_parameters(class)
    expected = nas_ep_random_numbers(p)
    config.N == expected || error("NAS EP class $class requires N=$expected")
    available = JACC.Multi.ndev()
    available == config.gpus || error(
        "JACC sees $available GPU(s), but this run requested $(config.gpus)"
    )
    batches = nas_ep_batches(p)
    batches % config.gpus == 0 || error("NAS EP streams must divide the JACC GPU count")
    return JACCNASEP(class, config.N, batches, config.gpus)
end

function model_initialize(b::JACCNASEP)
    return JACCNASEPState(
        JACC.Multi.array(zeros(Float64, NAS_EP_NQ, b.batches)),
        JACC.Multi.array(zeros(Float64, b.batches)),
        JACC.Multi.array(zeros(Float64, b.batches)),
    )
end

function jacc_nas_ep_kernel(i, q, sx, sy, part_length, jump)
    device = JACC.Multi.device_id(q)
    batch = (device - 1)*part_length + i - 1
    partial = nas_ep_batch(batch, jump)
    @inbounds begin
        q[1, i] = partial.q0
        q[2, i] = partial.q1
        q[3, i] = partial.q2
        q[4, i] = partial.q3
        q[5, i] = partial.q4
        q[6, i] = partial.q5
        q[7, i] = partial.q6
        q[8, i] = partial.q7
        q[9, i] = partial.q8
        q[10, i] = partial.q9
        sx[i] = partial.sx
        sy[i] = partial.sy
    end
    return nothing
end

function model_run!(b::JACCNASEP, s::JACCNASEPState)
    JACC.Multi.parallel_for(
        b.batches, jacc_nas_ep_kernel, s.q, s.sx, s.sy,
        b.batches ÷ b.gpus, nas_ep_batch_jump(),
    )
    return s
end

# Multi.parallel_for synchronizes participating devices before returning.
model_synchronize(::JACCNASEP) = nothing
model_throughput_label(::JACCNASEP) = "G random numbers/s"

function model_check_correctness(b::JACCNASEP, config)
    state = model_initialize(b)
    model_run!(b, state)
    sx, sy = sum(JACC.to_host(state.sx)), sum(JACC.to_host(state.sy))
    return nas_ep_verified(b.class, sx, sy) ? "pass" : "fail"
end

function model_correctness_context(b::JACCNASEP, config)
    return (; reference="NPB-GPU", dims=(b.N, 1))
end
