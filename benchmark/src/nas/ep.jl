const NAS_EP_NPB_GPU_COMMIT = "3f12d84920ee315ab00ef283717c1e74b68f4d00"
const NAS_EP_MK = 8
const NAS_EP_NQ = 10
const NAS_EP_SEED = 271828183.0
const NAS_EP_MULTIPLIER = 1220703125.0
const NAS_EP_EPSILON = 1.0e-8

const NAS_EP_CLASSES = Dict(
    "S" => (; m=24, sx=-3.247834652034740e3, sy=-6.958407078382297e3),
    "W" => (; m=25, sx=-2.863319731645753e3, sy=-6.320053679109499e3),
    "A" => (; m=28, sx=-4.295875165629892e3, sy=-1.580732573678431e4),
    "B" => (; m=30, sx=4.033815542441498e4, sy=-2.660669192809235e4),
    "C" => (; m=32, sx=4.764367927995374e4, sy=-8.084072988043731e4),
    "D" => (; m=36, sx=1.982481200946593e5, sy=-1.020596636361769e5),
    "E" => (; m=40, sx=-5.319717441530e5, sy=-3.688834557731e5),
)

function nas_ep_parameters(class::AbstractString)
    key = uppercase(class)
    return get(NAS_EP_CLASSES, key) do
        known = join(sort!(collect(keys(NAS_EP_CLASSES))), ", ")
        return error("Unknown NAS EP class '$class'; expected one of $known")
    end
end

nas_ep_batches(p) = Int(1) << (p.m - NAS_EP_MK)
nas_ep_pairs_per_batch() = Int(1) << NAS_EP_MK
nas_ep_random_numbers(p) = Int(1) << (p.m + 1)

@inline function nas_ep_randlc(x::Float64, a::Float64=NAS_EP_MULTIPLIER)
    r23, t23 = 2.0^-23, 2.0^23
    r46, t46 = 2.0^-46, 2.0^46
    a1 = trunc(r23*a)
    a2 = a - t23*a1
    x1 = trunc(r23*x)
    x2 = x - t23*x1
    t1 = a1*x2 + a2*x1
    z = t1 - t23*trunc(r23*t1)
    t3 = t23*z + a2*x2
    next = t3 - t46*trunc(r46*t3)
    return next, r46*next
end

function nas_ep_ipow46(a::Float64, exponent::Integer)
    exponent == 0 && return 1.0
    q, r, n = a, 1.0, Int(exponent)
    while n > 1
        n2 = n ÷ 2
        if 2n2 == n
            q, _ = nas_ep_randlc(q, q)
            n = n2
        else
            r, _ = nas_ep_randlc(r, q)
            n -= 1
        end
    end
    return first(nas_ep_randlc(r, q))
end

nas_ep_batch_jump() = nas_ep_ipow46(NAS_EP_MULTIPLIER, 2nas_ep_pairs_per_batch())

struct NASEPPartial
    q0::Float64
    q1::Float64
    q2::Float64
    q3::Float64
    q4::Float64
    q5::Float64
    q6::Float64
    q7::Float64
    q8::Float64
    q9::Float64
    sx::Float64
    sy::Float64
end

NASEPPartial() = NASEPPartial(ntuple(_ -> 0.0, 12)...)

@inline function nas_ep_start_seed(batch::Integer, jump::Float64)
    seed, power, k = NAS_EP_SEED, jump, batch
    while true
        half = k ÷ 2
        if 2half != k
            seed, _ = nas_ep_randlc(seed, power)
        end
        half == 0 && return seed
        power, _ = nas_ep_randlc(power, power)
        k = half
    end
end

@inline function nas_ep_batch(batch::Integer, jump::Float64=nas_ep_batch_jump())
    seed = nas_ep_start_seed(batch, jump)
    q0 = q1 = q2 = q3 = q4 = q5 = q6 = q7 = q8 = q9 = 0.0
    sx = sy = 0.0
    for _ in 1:nas_ep_pairs_per_batch()
        seed, u1 = nas_ep_randlc(seed)
        seed, u2 = nas_ep_randlc(seed)
        x1, x2 = 2u1 - 1.0, 2u2 - 1.0
        radius = x1*x1 + x2*x2
        if radius <= 1.0
            scale = sqrt(-2.0*log(radius)/radius)
            g1, g2 = x1*scale, x2*scale
            bin = floor(Int, max(abs(g1), abs(g2)))
            if bin == 0
                q0 += 1
            elseif bin == 1
                q1 += 1
            elseif bin == 2
                q2 += 1
            elseif bin == 3
                q3 += 1
            elseif bin == 4
                q4 += 1
            elseif bin == 5
                q5 += 1
            elseif bin == 6
                q6 += 1
            elseif bin == 7
                q7 += 1
            elseif bin == 8
                q8 += 1
            elseif bin == 9
                q9 += 1
            end
            sx += g1
            sy += g2
        end
    end
    return NASEPPartial(q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, sx, sy)
end

nas_ep_q(p::NASEPPartial) = ntuple(i -> getfield(p, i), NAS_EP_NQ)

function nas_ep_combine(partials)
    q = zeros(Float64, NAS_EP_NQ)
    sx = sy = 0.0
    for partial in partials
        for i in 1:NAS_EP_NQ
            q[i] += getfield(partial, i)
        end
        sx += partial.sx
        sy += partial.sy
    end
    return (; q, sx, sy)
end

function nas_ep_verified(class::AbstractString, sx::Real, sy::Real)
    p = nas_ep_parameters(class)
    return abs((sx - p.sx)/p.sx) <= NAS_EP_EPSILON &&
           abs((sy - p.sy)/p.sy) <= NAS_EP_EPSILON
end
