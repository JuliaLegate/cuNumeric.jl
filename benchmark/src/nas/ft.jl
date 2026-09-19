const NAS_FT_NPB_GPU_COMMIT = "3f12d84920ee315ab00ef283717c1e74b68f4d00"
const NAS_FT_SEED = 314159265.0
const NAS_FT_MULTIPLIER = 1220703125.0
const NAS_FT_ALPHA = 1.0e-6
const NAS_FT_CHECKSUM_SAMPLES = 1024

const NAS_FT_CLASSES = Dict(
    "S" => (; nx=64, ny=64, nz=64, niter=6),
    "W" => (; nx=128, ny=128, nz=32, niter=6),
    "A" => (; nx=256, ny=256, nz=128, niter=6),
    "B" => (; nx=512, ny=256, nz=256, niter=20),
    "C" => (; nx=512, ny=512, nz=512, niter=20),
    "D" => (; nx=2048, ny=1024, nz=1024, niter=25),
    "E" => (; nx=4096, ny=2048, nz=2048, niter=25),
)

# CUDA/FT/ft.cu::verify at NAS_FT_NPB_GPU_COMMIT.
const NAS_FT_CHECKSUMS = Dict(
    "S" => ComplexF64[
        5.546087004964e2 + 4.845363331978e2im,
        5.546385409189e2 + 4.865304269511e2im,
        5.546148406171e2 + 4.883910722336e2im,
        5.545423607415e2 + 4.901273169046e2im,
        5.544255039624e2 + 4.917475857993e2im,
        5.542683411902e2 + 4.932597244941e2im,
    ],
    "W" => ComplexF64[
        5.673612178944e2 + 5.293246849175e2im,
        5.631436885271e2 + 5.282149986629e2im,
        5.594024089970e2 + 5.270996558037e2im,
        5.560698047020e2 + 5.260027904925e2im,
        5.530898991250e2 + 5.249400845633e2im,
        5.504159734538e2 + 5.239212247086e2im,
    ],
    "A" => ComplexF64[
        5.046735008193e2 + 5.114047905510e2im,
        5.059412319734e2 + 5.098809666433e2im,
        5.069376896287e2 + 5.098144042213e2im,
        5.077892868474e2 + 5.101336130759e2im,
        5.085233095391e2 + 5.104914655194e2im,
        5.091487099959e2 + 5.107917842803e2im,
    ],
    "B" => ComplexF64[
        5.177643571579e2 + 5.077803458597e2im,
        5.154521291263e2 + 5.088249431599e2im,
        5.146409228649e2 + 5.096208912659e2im,
        5.142378756213e2 + 5.101023387619e2im,
        5.139626667737e2 + 5.103976610617e2im,
        5.137423460082e2 + 5.105948019802e2im,
        5.135547056878e2 + 5.107404165783e2im,
        5.133910925466e2 + 5.108576573661e2im,
        5.132470705390e2 + 5.109577278523e2im,
        5.131197729984e2 + 5.110460304483e2im,
        5.130070319283e2 + 5.111252433800e2im,
        5.129070537032e2 + 5.111968077718e2im,
        5.128182883502e2 + 5.112616233064e2im,
        5.127393733383e2 + 5.113203605551e2im,
        5.126691062020e2 + 5.113735928093e2im,
        5.126064276004e2 + 5.114218460548e2im,
        5.125504076570e2 + 5.114656139760e2im,
        5.125002331720e2 + 5.115053595966e2im,
        5.124551951846e2 + 5.115415130407e2im,
        5.124146770029e2 + 5.115744692211e2im,
    ],
    "C" => ComplexF64[
        5.195078707457e2 + 5.149019699238e2im,
        5.155422171134e2 + 5.127578201997e2im,
        5.144678022222e2 + 5.122251847514e2im,
        5.140150594328e2 + 5.121090289018e2im,
        5.137550426810e2 + 5.121143685824e2im,
        5.135811056728e2 + 5.121496764568e2im,
        5.134569343165e2 + 5.121870921893e2im,
        5.133651975661e2 + 5.122193250322e2im,
        5.132955192805e2 + 5.122454735794e2im,
        5.132410471738e2 + 5.122663649603e2im,
        5.131971141679e2 + 5.122830879827e2im,
        5.131605205716e2 + 5.122965869718e2im,
        5.131290734194e2 + 5.123075927445e2im,
        5.131012720314e2 + 5.123166486553e2im,
        5.130760908195e2 + 5.123241541685e2im,
        5.130528295923e2 + 5.123304037599e2im,
        5.130310107773e2 + 5.123356167976e2im,
        5.130103090133e2 + 5.123399592211e2im,
        5.129905029333e2 + 5.123435588985e2im,
        5.129714421109e2 + 5.123465164008e2im,
    ],
    "D" => ComplexF64[
        5.122230065252e2 + 5.118534037109e2im,
        5.120463975765e2 + 5.117061181082e2im,
        5.119865766760e2 + 5.117096364601e2im,
        5.119518799488e2 + 5.117373863950e2im,
        5.119269088223e2 + 5.117680347632e2im,
        5.119082416858e2 + 5.117967875532e2im,
        5.118943814638e2 + 5.118225281841e2im,
        5.118842385057e2 + 5.118451629348e2im,
        5.118769435632e2 + 5.118649119387e2im,
        5.118718203448e2 + 5.118820803844e2im,
        5.118683569061e2 + 5.118969781011e2im,
        5.118661708593e2 + 5.119098918835e2im,
        5.118649768950e2 + 5.119210777066e2im,
        5.118645605626e2 + 5.119307604484e2im,
        5.118647586618e2 + 5.119391362671e2im,
        5.118654451572e2 + 5.119463757241e2im,
        5.118665212451e2 + 5.119526269238e2im,
        5.118679083821e2 + 5.119580184108e2im,
        5.118695433664e2 + 5.119626617538e2im,
        5.118713748264e2 + 5.119666538138e2im,
        5.118733606701e2 + 5.119700787219e2im,
        5.118754661974e2 + 5.119730095953e2im,
        5.118776626738e2 + 5.119755100241e2im,
        5.118799262314e2 + 5.119776353561e2im,
        5.118822370068e2 + 5.119794338060e2im,
    ],
    "E" => ComplexF64[
        5.121601045346e2 + 5.117395998266e2im,
        5.120905403678e2 + 5.118614716182e2im,
        5.120623229306e2 + 5.119074203747e2im,
        5.120438418997e2 + 5.119345900733e2im,
        5.120311521872e2 + 5.119551325550e2im,
        5.120226088809e2 + 5.119720179919e2im,
        5.120169296534e2 + 5.119861371665e2im,
        5.120131225172e2 + 5.119979364402e2im,
        5.120104767108e2 + 5.120077674092e2im,
        5.120085127969e2 + 5.120159443121e2im,
        5.120069224127e2 + 5.120227453670e2im,
        5.120055158164e2 + 5.120284096041e2im,
        5.120041820159e2 + 5.120331373793e2im,
        5.120028605402e2 + 5.120370938679e2im,
        5.120015223011e2 + 5.120404138831e2im,
        5.120001570022e2 + 5.120432068837e2im,
        5.119987650555e2 + 5.120455615860e2im,
        5.119973525091e2 + 5.120475499442e2im,
        5.119959279472e2 + 5.120492304629e2im,
        5.119945006558e2 + 5.120506508902e2im,
        5.119930795911e2 + 5.120518503782e2im,
        5.119916728462e2 + 5.120528612016e2im,
        5.119902874185e2 + 5.120537101195e2im,
        5.119889291565e2 + 5.120544194514e2im,
        5.119876028049e2 + 5.120550079284e2im,
    ],
)

function nas_ft_parameters(class::AbstractString)
    key = uppercase(class)
    return get(NAS_FT_CLASSES, key) do
        known = join(sort!(collect(keys(NAS_FT_CLASSES))), ", ")
        return error("Unknown NAS FT class '$class'; expected one of $known")
    end
end

function nas_ft_randlc(x::Float64, a::Float64=NAS_FT_MULTIPLIER)
    r23, t23 = 2.0^-23, 2.0^23
    r46, t46 = 2.0^-46, 2.0^46
    t1 = r23*a
    a1 = trunc(Int, t1)
    a2 = a - t23*a1
    t1 = r23*x
    x1 = trunc(Int, t1)
    x2 = x - t23*x1
    t1 = a1*x2 + a2*x1
    t2 = trunc(Int, r23*t1)
    z = t1 - t23*t2
    t3 = t23*z + a2*x2
    t4 = trunc(Int, r46*t3)
    next = t3 - t46*t4
    return next, r46*next
end

function nas_ft_ipow46(a::Float64, exponent::Integer)
    exponent == 0 && return 1.0
    q, r, n = a, 1.0, Int(exponent)
    while n > 1
        n2 = n ÷ 2
        if 2n2 == n
            q, _ = nas_ft_randlc(q, q)
            n = n2
        else
            r, _ = nas_ft_randlc(r, q)
            n -= 1
        end
    end
    r, _ = nas_ft_randlc(r, q)
    return r
end

function nas_ft_plane_starts!(starts::Vector{Float64}, nx::Integer, ny::Integer)
    jump = nas_ft_ipow46(NAS_FT_MULTIPLIER, 2nx*ny)
    start = NAS_FT_SEED
    @inbounds for k in eachindex(starts)
        starts[k] = start
        start, _ = nas_ft_randlc(start, jump)
    end
    return starts
end

"""Generate NPB FT's complex initial field in its exact RNG order."""
function nas_ft_initial_conditions!(out::Array{ComplexF64,3})
    nx, ny, nz = size(out)
    starts = nas_ft_plane_starts!(Vector{Float64}(undef, nz), nx, ny)
    @inbounds for k in 1:nz
        x = starts[k]
        for j in 1:ny, i in 1:nx
            x, realpart = nas_ft_randlc(x)
            x, imagpart = nas_ft_randlc(x)
            out[i, j, k] = ComplexF64(realpart, imagpart)
        end
    end
    return out
end

function nas_ft_initial_conditions(p)
    return nas_ft_initial_conditions!(Array{ComplexF64}(undef, p.nx, p.ny, p.nz))
end

function nas_ft_twiddle!(twiddle::Array{Float64,3})
    nx, ny, nz = size(twiddle)
    ap = -4.0*NAS_FT_ALPHA*pi^2
    @inbounds for k in 0:(nz - 1), j in 0:(ny - 1), i in 0:(nx - 1)
        kk = mod(k + nz÷2, nz) - nz÷2
        jj = mod(j + ny÷2, ny) - ny÷2
        ii = mod(i + nx÷2, nx) - nx÷2
        twiddle[i + 1, j + 1, k + 1] = exp(ap*(ii^2 + jj^2 + kk^2))
    end
    return twiddle
end

function nas_ft_twiddle(p)
    return nas_ft_twiddle!(Array{Float64}(undef, p.nx, p.ny, p.nz))
end

function nas_ft_checksum_indices(p)
    linear = LinearIndices((p.nx, p.ny, p.nz))
    return [
        linear[mod(j, p.nx) + 1, mod(3j, p.ny) + 1, mod(5j, p.nz) + 1]
        for j in 1:NAS_FT_CHECKSUM_SAMPLES
    ]
end

function nas_ft_checksum_mask(p)
    mask = zeros(Float64, p.nx, p.ny, p.nz)
    @inbounds for index in nas_ft_checksum_indices(p)
        mask[index] += 1.0
    end
    return mask
end

function nas_ft_verified(class::AbstractString, got; tolerance=1.0e-12)
    expected = NAS_FT_CHECKSUMS[uppercase(class)]
    length(got) == length(expected) || return false
    return all(
        abs((actual - reference)/reference) <= tolerance for
        (actual, reference) in zip(got, expected)
    )
end
