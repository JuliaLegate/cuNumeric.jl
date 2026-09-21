# Investigation-only adapter: run IterativeSolvers' CG source in a separate
# module, with scalar extraction at the solver's reduction boundary. This does
# not change cuNumeric's API or overwrite methods in IterativeSolvers.
# Uses private helpers/source layout from the pinned IterativeSolvers 0.9.4.
module ScalarCG
import cuNumeric
import LinearAlgebra
import IterativeSolvers
using LinearAlgebra: mul!, ldiv!
using IterativeSolvers: Identity, zerox, ConvergenceHistory, reserve!, nextiter!, setconv, shrink!

norm(x::cuNumeric.NDArray) = only(LinearAlgebra.norm(x))
dot(x::cuNumeric.NDArray, y::cuNumeric.NDArray) = only(LinearAlgebra.dot(x, y))

include(joinpath(pkgdir(IterativeSolvers), "src", "cg.jl"))
end
