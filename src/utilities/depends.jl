using Preferences
using CNPreferences: CNPreferences
using Legate
using OpenSSL_jll
using OpenBLAS32_jll
using Libdl
using CxxWrap
using Pkg
using TOML

using cupynumeric_jll
# temporary until cunumeric_jl_wrapper_jll exists
# using cunumeric_jl_wrapper_jll
using CUTENSOR_jll

using LinearAlgebra
import LinearAlgebra: mul!

using Random
import Random: rand, rand!

import Base: abs, angle, acos, acosh, asin, asinh, atan, atanh, cbrt,
    ceil, clamp, conj, cos, cosh, cosh, deg2rad, exp, exp2, expm1,
    floor, frexp, imag, isfinite, isinf, isnan, log, log10,
    log1p, log2, !, modf, -, rad2deg, sign, signbit, sin,
    sinh, sqrt, tan, tanh, trunc, +, *, atan, &, |, ⊻, copysign,
    /, ==, ^, div, gcd, >, >=, hypot, isapprox, lcm, ldexp, <<,
    <, <=, !=, >>, all, any, argmax, argmin, maximum, minimum,
    prod, sum, read, trues, falses, axes, view
