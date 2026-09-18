"""NAS FT using cuPyNumeric's full 3-D FFT.

LIMITATION: cuPyNumeric has no NPB 46-bit RNG primitive, so the exact initial
field is generated on the host during every timed run and copied to Legate.
The full 3-D FFT is a native Legate auto task and distributes across available
GPUs. The 1024-point checksum is currently a full masked reduction because
cuPyNumeric has no indexed-reduction primitive. Official verification is kept.
"""

import math
import cupynumeric as np
import numpy as host_np
from core import register_benchmark

SEED, MULTIPLIER, ALPHA = 314159265.0, 1220703125.0, 1.0e-6
CHECKSUM_SAMPLES = 1024
CLASSES = {
    "S": (64, 64, 64, 6), "W": (128, 128, 32, 6),
    "A": (256, 256, 128, 6), "B": (512, 256, 256, 20),
    "C": (512, 512, 512, 20), "D": (2048, 1024, 1024, 25),
    "E": (4096, 2048, 2048, 25),
}
CHECKSUMS = {
    "S": [
        554.6087004964+484.5363331978j, 554.6385409189+486.5304269511j,
        554.6148406171+488.3910722336j, 554.5423607415+490.1273169046j,
        554.4255039624+491.7475857993j, 554.2683411902+493.2597244941j,
    ],
    "W": [
        567.3612178944+529.3246849175j, 563.1436885271+528.2149986629j,
        559.4024089970+527.0996558037j, 556.0698047020+526.0027904925j,
        553.0898991250+524.9400845633j, 550.4159734538+523.9212247086j,
    ],
    "A": [
        504.6735008193+511.4047905510j, 505.9412319734+509.8809666433j,
        506.9376896287+509.8144042213j, 507.7892868474+510.1336130759j,
        508.5233095391+510.4914655194j, 509.1487099959+510.7917842803j,
    ],
    "B": [
        517.7643571579+507.7803458597j, 515.4521291263+508.8249431599j,
        514.6409228649+509.6208912659j, 514.2378756213+510.1023387619j,
        513.9626667737+510.3976610617j, 513.7423460082+510.5948019802j,
        513.5547056878+510.7404165783j, 513.3910925466+510.8576573661j,
        513.2470705390+510.9577278523j, 513.1197729984+511.0460304483j,
        513.0070319283+511.1252433800j, 512.9070537032+511.1968077718j,
        512.8182883502+511.2616233064j, 512.7393733383+511.3203605551j,
        512.6691062020+511.3735928093j, 512.6064276004+511.4218468548j,
        512.5504076570+511.4656139760j, 512.5002331720+511.5053595966j,
        512.4551951846+511.5415130407j, 512.4146770029+511.5744692211j,
    ],
    "C": [
        519.5078707457+514.9019699238j, 515.5422171134+512.7578201997j,
        514.4678022222+512.2251847514j, 514.0150594328+512.1090289018j,
        513.7550426810+512.1143685824j, 513.5811056728+512.1496764568j,
        513.4569343165+512.1870921893j, 513.3651975661+512.2193250322j,
        513.2955192805+512.2454735794j, 513.2410471738+512.2663649603j,
        513.1971141679+512.2830879827j, 513.1605205716+512.2965869718j,
        513.1290734194+512.3075927445j, 513.1012720314+512.3166486553j,
        513.0760908195+512.3241541685j, 513.0528295923+512.3304037599j,
        513.0310107773+512.3356167976j, 513.0103090133+512.3399592211j,
        512.9905029333+512.3435588985j, 512.9714421109+512.3465164008j,
    ],
    "D": [
        512.2230065252+511.8534037109j, 512.0463975765+511.7061181082j,
        511.9865766760+511.7096364601j, 511.9518799488+511.7373863950j,
        511.9269088223+511.7680347632j, 511.9082416858+511.7967875532j,
        511.8943814638+511.8225281841j, 511.8842385057+511.8451629348j,
        511.8769435632+511.8649119387j, 511.8718203448+511.8820803844j,
        511.8683569061+511.8969781011j, 511.8661708593+511.9098918835j,
        511.8649768950+511.9210777066j, 511.8645605626+511.9307604484j,
        511.8647586618+511.9391362671j, 511.8654451572+511.9463757241j,
        511.8665212451+511.9526269238j, 511.8679083821+511.9580184108j,
        511.8695433664+511.9626617538j, 511.8713748264+511.9666538138j,
        511.8733606701+511.9700787219j, 511.8754661974+511.9730095953j,
        511.8776626738+511.9755100241j, 511.8799262314+511.9776353561j,
        511.8822370068+511.9794338060j,
    ],
    "E": [
        512.1601045346+511.7395998266j, 512.0905403678+511.8614716182j,
        512.0623229306+511.9074203747j, 512.0438418997+511.9345900733j,
        512.0311521872+511.9551325550j, 512.0226088809+511.9720179919j,
        512.0169296534+511.9861371665j, 512.0131225172+511.9979364402j,
        512.0104767108+512.0077674092j, 512.0085127969+512.0159443121j,
        512.0069224127+512.0227453670j, 512.0055158164+512.0284096041j,
        512.0041820159+512.0331373793j, 512.0028605402+512.0370938679j,
        512.0015223011+512.0404138831j, 512.0001570022+512.0432068837j,
        511.9987650555+512.0455615860j, 511.9973525091+512.0475499442j,
        511.9959279472+512.0492304629j, 511.9945006558+512.0506508902j,
        511.9930795911+512.0518503782j, 511.9916728462+512.0528612016j,
        511.9902874185+512.0537101195j, 511.9889291565+512.0544194514j,
        511.9876028049+512.0550079284j,
    ],
}

def randlc(x, a=MULTIPLIER):
    r23, t23, r46, t46 = 2.0**-23, 2.0**23, 2.0**-46, 2.0**46
    a1 = int(r23*a); a2 = a-t23*a1
    x1 = int(r23*x); x2 = x-t23*x1
    t1 = a1*x2+a2*x1; z = t1-t23*int(r23*t1)
    t3 = t23*z+a2*x2; x = t3-t46*int(r46*t3)
    return x, r46*x

def ipow46(a, exponent):
    if exponent == 0:
        return 1.0
    q, r, n = a, 1.0, exponent
    while n > 1:
        n2 = n//2
        if 2*n2 == n:
            q, _ = randlc(q, q); n = n2
        else:
            r, _ = randlc(r, q); n -= 1
    return randlc(r, q)[0]

def initial_conditions(out):
    nz, ny, nx = out.shape
    jump, start, flat, plane = ipow46(MULTIPLIER, 2*nx*ny), SEED, out.reshape(-1), nx*ny
    for k in range(nz):
        x = start
        for i in range(plane):
            x, re = randlc(x); x, im = randlc(x)
            flat[k*plane+i] = re + 1j*im
        start, _ = randlc(start, jump)

def frequency_squares(n):
    return host_np.asarray([((i+n//2) % n-n//2)**2 for i in range(n)])

class NASFourierTransform:
    name = "nas_ft"
    def __init__(self, T, N, M, **kwargs):
        self.T, self.N, self.M = T, N, M
        self.class_name = str(kwargs.pop("class", "S")).upper()
        if kwargs:
            raise ValueError(f"Unknown NAS FT options: {', '.join(kwargs)}")
        nx, ny, _, _ = CLASSES[self.class_name]
        if T is not np.float64 or (N, M) != (nx, ny):
            raise ValueError(f"NAS FT class {self.class_name} requires Float64, N={nx}, M={ny}")
    def dims(self):
        return self.N, self.M
    def initialize(self):
        nx, ny, nz, _ = CLASSES[self.class_name]; shape = (nz, ny, nx)
        mask = host_np.zeros(shape)
        for j in range(1, CHECKSUM_SAMPLES+1):
            mask[(5*j) % nz, (3*j) % ny, j % nx] += 1.0
        return {
            "mask": np.asarray(mask),
            "ix2": np.asarray(frequency_squares(nx)).reshape(1, 1, nx),
            "iy2": np.asarray(frequency_squares(ny)).reshape(1, ny, 1),
            "iz2": np.asarray(frequency_squares(nz)).reshape(nz, 1, 1),
            "host_initial": host_np.empty(shape, dtype=host_np.complex128),
        }
    def run(self, state):
        niter = CLASSES[self.class_name][3]
        initial_conditions(state["host_initial"])
        u0 = np.asarray(state["host_initial"])
        twiddle = np.exp((-4.0*ALPHA*math.pi**2) *
            (state["ix2"]+state["iy2"]+state["iz2"]))
        u0, checksums = np.fft.fftn(u0), []
        for _ in range(niter):
            u0 *= twiddle
            checksums.append(np.sum(np.fft.ifftn(u0)*state["mask"]))
        return checksums
    def correctness_dims(self):
        return self.N, self.M
    def check_correctness(self):
        got = [complex(host_np.asarray(x)) for x in self.run(self.initialize())]
        ok = all(abs((x-r)/r) <= 1.0e-12 for x, r in zip(got, CHECKSUMS[self.class_name]))
        return "pass" if ok else "fail"

register_benchmark("nas_ft", NASFourierTransform)
