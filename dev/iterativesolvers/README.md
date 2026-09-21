# IterativeSolvers investigation

Environment on dubliner: `/pool/emeitz/iterativesolvers-investigation-20260920`.
It develops `/pool/emeitz/repos/cuNumeric.jl` and installs IterativeSolvers 0.9.4.
The original checkout was `c4157d15634df8c1472faa883e33acab62ab749e`; changes are
Julia-only. No C/C++ wrapper changes or rebuild are needed.

## Reproduce

From a Linux cuNumeric checkout, create a separate environment:

```sh
mkdir -p /path/to/solver-environment
# If the checkout uses local library preferences, copy those too:
cp LocalPreferences.toml /path/to/solver-environment/
julia --project=/path/to/solver-environment dev/iterativesolvers/setup.jl
```

On dubliner the test command is:

```sh
export JULIA_DEPOT_PATH=/pool/emeitz/.julia:/home/emeitz/.julia
export LEGATE_AUTO_CONFIG=0
export LEGATE_CONFIG='--gpus 1 --cpus 2 --fbmem 2048 --sysmem 2048 --zcmem 256'
~/.juliaup/bin/julia --startup-file=no \
  --project=/pool/emeitz/iterativesolvers-investigation-20260920 \
  /pool/emeitz/repos/cuNumeric.jl/dev/iterativesolvers/investigate.jl
```

The memory cap leaves room for the other GPU processes. Run with default Julia
threading to exercise cleanup with both thread pools present.

## Physical-handle cleanup regression

The original default-threading abort was real: `Array(x)` left temporary Legate
PhysicalArray/PhysicalStore handles for GC via `cuNumeric.get_ptr`. Destroying
those handles on another Julia thread could call `unmap_region` outside the
runtime task context. NDArray's existing deferred-free queue did not cover them.

A minimal CPU-only reproduction (Julia 1.13.1-DEV.3, main task in the interactive
pool) failed on its first collection before the fix and passed ten rounds after:

```julia
using cuNumeric, Test
function copy_once()
    x = cuNumeric.ones(Float64, 8)
    @test Array(x) == ones(8)
    cuNumeric.destroy!(x)
end
for _ in 1:10
    copy_once()
    fetch(Threads.@spawn :default GC.gc(true))
    cuNumeric.drain_pending_frees!()
end
```

`get_ptr` now explicitly releases both physical handles on the runtime thread.
No wrapper change is needed. `test/analysis/lifetime.jl` contains the regression
and selects the thread pool opposite to the caller. The separate environment
preserves `repro-copy-gc-before.log` and `repro-copy-gc-after.log`.

## Current status

The NDArray interfaces now provide matrix-vector multiplication, five-argument
matrix-matrix multiplication, conjugating dot products, entrywise norms, vector
updates, and diagonal solves. `dot` and `norm` return asynchronous 0D NDArrays.
Norm uses unscaled mapped power accumulation followed by a backend root and
currently requires a GPU target. Integer-integer matrix multiplication is
unsupported; mixed numeric inputs follow the package's promotion policy.

Matrix-vector multiplication uses `contract!`, which selects cuPyNumeric's
specialized `MATVECMUL` task. No wrapper changes are needed.

IterativeSolvers CG expects Julia scalar coefficients and convergence values.
`scalar_cg.jl` loads the installed 0.9.4 CG source into a separate `ScalarCG`
module, supplying solver-local `norm` and `dot` that explicitly extract their
results with `only`. It does not change cuNumeric or overwrite IterativeSolvers
methods. This experiment depends on that version's private helpers/source layout.
It synchronizes at reductions; it is not an asynchronous CG implementation.

`investigate.jl` tests adapted CG and Jacobi PCG across the four floating-point
types, including nonzero initial guesses and a zero RHS. It compares true host
residuals and direct solves, with scalar indexing disabled. It also reports the
expected scalar-interface failure of unadapted CG. Earlier solver logs describe
the historical scalar-returning NDArray prototype.
Current interface tests live in `test/array/vector_linalg.jl` (operations,
promotion, and errors) and `test/gpu_only/vector_norm.jl` (norm correctness and
inference). The separate environment retains the investigation logs; no
IterativeSolvers dependency was added to cuNumeric itself.

## Validated after the cleanup fix

On dubliner with default Julia 1.13.1-DEV.3 threading and one GPU,
`validate-fix-cg.log` records 2,478 passing assertions: 56 lifetime checks,
2,038 vector/promotion/error checks, 316 norm correctness/inference checks,
and 68 adapted CG/PCG checks. The process exited successfully. Stock CG's
scalar-comparison failure is an expected diagnostic, not a passing solver test.
The full package suite and multi-GPU execution were not tested.

## Backend-coefficient PCG

`backend_pcg.jl` defines an investigation-only `BackendPCG.PCGIterable` and
specializes `IterativeSolvers.done` / `converged` for that type. It leaves
cuNumeric and the stock IterativeSolvers iterables unchanged.

```julia
include("/pool/emeitz/repos/cuNumeric.jl/dev/iterativesolvers/backend_pcg.jl")
it = BackendPCG.pcg(A, b; Pl=Diagonal(NDArray(diag(Ah))), reltol=1e-8)
x = it.x
IterativeSolvers.converged(it)
```

`rho`, `alpha`, `beta`, the residual norm, and the tolerance are 0D NDArrays.
Coefficient arithmetic and vector updates use backend broadcasts. Only `done`
extracts a value: the Boolean `residual <= tolerance`. It checks once before
each iteration and once after the last, and `converged` returns the cached flag.
The norm and dot calls themselves never fetch. Complex runs currently use the
existing `allowpromotion` opt-in, as in the scalar-baseline tests.

The iterable yields iteration numbers; it does not use IterativeSolvers' scalar
residual history or verbose printing. To retain backend residual history, copy
`it.residual` explicitly after a step. `pcg!` returns the iterable/state, whose
`x` is the supplied solution array. This experiment assumes an SPD/Hermitian
positive-definite system and compatible positive-definite preconditioner.

The backend-coefficient PCG checks passed 152/152 assertions on dubliner with default threading (all four floating-point types, identity/Jacobi preconditioning). The 68 scalar-baseline CG/PCG assertions also passed in the same run; see backend-pcg.log in the separate environment.
