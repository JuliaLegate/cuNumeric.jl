# Accelerated CG step for cuNumeric, generated from the shared recurrence in
# ../../benchmarks/cg.jl. The plain step and the solver loop live there so the
# CUDA.jl worker shares the exact same workload.
if CUNUMERIC_BENCH_RUNTIME
    let body = deepcopy(CG_STEP_BODY)
        signature = :(
            cg_step!(
                b::ConjugateGradientAccelerated{T}, x, r, p, Ap, lower, diagonal, upper, rho
            ) where {T}
        )
        @eval $(_define_accelerated_definition(signature, body))
    end
end
