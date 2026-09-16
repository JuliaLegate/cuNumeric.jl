# GPU mapped reductions

The frontend in `../ndarray/mapreduce.jl` selects reduction policies, validates
shapes/types, and handles empty inputs. `mapreduce.jl` compiles and caches Julia
PTX kernels; capture values and array extents are runtime arguments.

Each Legate GPU task:

1. Packs the input tile's lower-bound pointer, extents, and element strides.
   `_mr_offset` enumerates retained and reduced coordinates separately, so slices
   need not be contiguous.
2. Maps values and reduces them in 256-thread blocks using shared memory.
3. Combines block partials into the task's reduction store, with one writer per
   retained coordinate through an exclusive reduction accessor.

**Legate combines contributions between tasks and GPUs.** The native glue in
`../../lib/cunumeric_jl_wrapper/src/mapreduce.cpp` packs descriptors and launches
PTX on the task stream. Scratch is task-local and bounded to 4,096 partials;
there is no input-sized mapped temporary.

The reduction tasks use a separate Legate library with a mapper that reserves
64 KiB of framebuffer scratch per allocating task (4,096 × 16 bytes). The reduction
variant declares `has_allocations`; both variants declare that all device work
uses the task stream. Registering them with cuPyNumeric's mapper would omit this
scratch reservation and can abort even when device memory is available.

Floating extrema use ordered unsigned keys to preserve NaNs and signed zeros.
Boolean product/extrema use 0/1 bytes. A finishing kernel decodes storage and
applies `init` once when needed; otherwise the accumulator is returned directly.
Singletons use output privileges to avoid identity arithmetic, while dimensional
sums/products retain Base's zero/one seed.

Submission copies capture bytes into task-owned scalars. Temporary NDArray
handles are explicitly released after submission; native store handles use C++
scope ownership. Warm calls neither extract host results nor insert execution
fences. Kernel registration uses the shared PTX loader on cache misses.

## Diagnosing CI crashes

Run the `gpu_only/mapreduce` test with `CUNUMERIC_MAPREDUCE_TRACE=1` to log and
flush each comparison case and its construction, submission, result-access, and
cleanup stages. Add `CUNUMERIC_MAPREDUCE_SYNC=1` to fence after submission when
separating asynchronous task failures from host-side failures. These switches
only affect the tests. Preserve the full stderr and Legate log files from the
test directory; an unsymbolized signal backtrace alone may not identify the call.
