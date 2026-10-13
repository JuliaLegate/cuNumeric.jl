# Tuning

## Distributed solves and factorizations

The `A \ b`, `cuNumeric.solve`, `cholesky`, and `qr` APIs select
distributed tasks automatically. Selection follows cuPyNumeric 26.06:

| Operation | cuSolverMp cutoff | Internal block size |
| :--- | :--- | :--- |
| Square solve | dimension ≥ 2048 | 512 |
| Lower Cholesky | dimension ≥ 8192 | 2048 |
| Reduced QR | matrix contains ≥ 1048576 elements | 128 |

These defaults can be changed through `CNPreferences`. The loaded library
must support cuSolverMp and Legate must have more than one active GPU. The
library selects the algorithm. Legate handles placement, communication, and
redistribution. Configure resources before starting Julia, as described in
[Hardware Configuration](../configuration/hardware.md).

Ordinary solve and QR tasks remain the fallback. Cholesky also uses the tiled
POTRF/TRSM/SYRK/GEMM algorithm for eligible multi-processor configurations.
Below its partitioning cutoff this is a single tile. Batched operations still
distribute independent matrices, each of which must fit on one processor.
SVD and general eigendecomposition do not support distributed factorizations.

Results retain their existing Julia types, element promotion, and shapes.
Inputs are preserved. Kernel failures propagate when the runtime reports them.
Failed collectives are not retried using another algorithm.

## Tuning

Use [`CNPreferences.set_linalg!`](../api_preferences.md#linear-algebra) to set any
of the seven documented constants. `MIN_*_MATRIX_SIZE` controls when an operation
can select cuSolverMp. Solve and Cholesky use the row count. QR uses the number
of matrix elements. `MIN_SOLVE_TILE_SIZE`, `MIN_CHOLESKY_TILE_SIZE`, and
`QR_TILE_SIZE` set the solver block sizes. QR uses the same block size on both axes.

For the tiled Cholesky fallback, `MIN_CHOLESKY_MATRIX_SIZE` also sets the
single-tile cutoff. `MIN_CHOLESKY_TILE_SIZE` guides tile subdivision, and
`MAX_CHOLESKY_TILES_PER_PROC` limits the number of tiles per matrix axis relative
to the processor count. These are tuning heuristics, not memory limits.

Restart Julia after changing these preferences. The library capabilities and
configured resources are determined when the runtime starts.

```julia
using CNPreferences

CNPreferences.set_linalg!(; MIN_SOLVE_MATRIX_SIZE=4096, MIN_SOLVE_TILE_SIZE=512)
# Restart Julia to use the new settings.
```
