using Preferences
using CNPreferences
using LegatePreferences
using Legate
using Libdl
using CxxWrap
using Pkg
using TOML

import Base: axes, convert, copy, copyto!, inv, isfinite, sqrt, -, +, *, ==, !=,
    isapprox, read, view, maximum, minimum, prod, sum, getindex, setindex!,
    sum, prod

using LinearAlgebra
import LinearAlgebra: mul!

using Random
import Random: rand!

using StatsBase
import StatsBase: var, mean
