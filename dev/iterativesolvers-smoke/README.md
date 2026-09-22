# Stock IterativeSolvers smoke test

`cg.jl` calls stock CG with NDArrays inside `@allowautofetch` and checks the
returned solution against a host-computed residual. `other_solvers.jl` repeats
CG and tries three more solvers, printing their errors without adapting them.
Neither cuNumeric nor IterativeSolvers is modified.

Tested on dubliner's GPU with Julia 1.12.7, IterativeSolvers 0.9.4, and
`codex/ndscalar-autounwrap` at `e0e2f7fd2c962c3aaf48454d7701bd2c3189c5dd`.
The input is a 32-by-32 Float64 SPD tridiagonal matrix with an all-ones RHS.

| Solver | Observed result |
| --- | --- |
| CG | Pass; relative residual `2.2633986948989007e-16`. |
| MINRES | `MethodError` constructing `MINRESIterable`: work vectors must be `DenseVector`; its two small vectors also have incompatible types (`Vector{Float64}` versus `Vector{CNFloat{Float64}}`). |
| GMRES | Scalar-indexing error in `copyto!(first_col, b)` at `gmres.jl:241`: the basis is a host `Matrix`. |
| BiCGStab(l) | Scalar-indexing error in `copyto!(residual, b)` at `bicgstabl.jl:47`: the workspace is a host `Matrix`. |

## Run on dubliner

The fresh environment and full logs are in
`/pool/emeitz/iterativesolvers-smoke.wZrnP4` (`cg.log`, `other_solvers.log`).

```sh
cd /pool/emeitz/iterativesolvers-smoke.wZrnP4
export JULIA_DEPOT_PATH=/pool/emeitz/.julia:/home/emeitz/.julia
export LEGATE_AUTO_CONFIG=0
export LEGATE_CONFIG='--gpus 1 --cpus 2 --fbmem 2048 --sysmem 2048 --zcmem 256'
/home/emeitz/.julia/juliaup/julia-1.12.7+0.x64.linux.gnu/bin/julia --startup-file=no --project=. cg.jl
# Replace cg.jl with other_solvers.jl to reproduce the other failures.
```

The supplied Project.toml selects the scalar Git branch through `[sources]`.
On this machine, Pkg's URL checkout failed with
`GitError(Code:ERROR, Class:Submodule, cannot get submodules without a working tree)`.
The tested environment instead uses `cuNumeric = {path = "cuNumeric.jl"}` pointing
to a fresh, unmodified Git clone of that exact branch and revision. Its backend
library preferences were copied from the working dubliner test environment.

## Krylov and KrylovKit

Run `krylov.jl` in the same environment (log: `krylov.log`). Tested releases:
Krylov 0.10.10, KrylovKit 0.10.4, VectorInterface 0.6.1, CUDACore 6.4.0.
The script compares NDArrays with CUDA CuArrays on the same SPD system, with
scalar indexing disabled and `@allowautofetch` enabled. No library methods are added.

| Call | NDArray | CuArray |
| --- | --- | --- |
| `Krylov.cg(A, b)` | Missing `NDArray{Float64,1,Nothing}(undef, n)` constructor at `krylov_workspaces.jl:273`. | Pass, residual `2.23e-16`. |
| `Krylov.cg!` with `CgWorkspace(KrylovConstructor(b))` | Gets past allocation; fails at `cg.jl:239` because `kaxpy!` requires matching coefficient and vector element types (`CNFloat{Float64}` versus `Float64`). | Pass, residual `2.23e-16`. |
| `KrylovKit.linsolve(A, b, zero(b), CG(...))` | Fails at `linsolve/cg.jl:4`: VectorInterface's `inner` calls two-input `mapreduce`, which cuNumeric does not support. | Pass, residual `2.39e-16`. |

Both libraries explicitly support CUDA arrays:
[Krylov GPU support](https://jso.dev/Krylov.jl/stable/gpu/) and
[KrylovKit vector support](https://jutho.github.io/KrylovKit.jl/stable/).
Krylov's documented custom workspace bypasses the constructor gap but does not
resolve its scalar dispatch requirements. KrylovKit's first integration point
is `VectorInterface.inner`; later compatibility has not been established.
