module cuNumericIterativeSolversExt
using cuNumeric, LinearAlgebra
import IterativeSolvers as IS

# Released 0.9.4 cannot dispatch its iteration methods on extension-owned types.
# Keep loading either package safe until the upstream iterator hooks are present.
const has_iterator_hooks = isdefined(IS, :AbstractCGIterable) &&
    isdefined(IS, :AbstractPCGIterable) && isdefined(IS, :cg_history_type) &&
    isdefined(IS, :cg_check_verbose)

if has_iterator_hooks
    include("iterativesolvers/cg.jl")
else
    function IS.cg_iterator!(x::NDArray, A, b::NDArray, Pl=IS.Identity(); kwargs...)
        throw(ArgumentError("NDArray CG requires the IterativeSolvers iterator hooks; " *
            "released IterativeSolvers 0.9.4 does not provide them. " *
            "See dev/iterativesolvers/IterativeSolvers-cg.patch and its README."))
    end
end
end
