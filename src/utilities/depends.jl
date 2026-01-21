using Preferences
using CNPreferences
import LegatePreferences
using Legate
using Libdl
using CxxWrap
using Pkg
using TOML

using cupynumeric_jll
using cunumeric_jl_wrapper_jll

import Base: axes, convert, copy, copyto!, inv, isfinite, sqrt, -, +, *, ==, !=,
    isapprox, read, view, maximum, minimum, prod, sum, getindex, setindex!,
    sum, prod

using LinearAlgebra
import LinearAlgebra: mul!

using Random
import Random: rand!

using StatsBase
import StatsBase: var, mean
