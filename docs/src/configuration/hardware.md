# Hardware Configuration

Legate chooses the hardware configuration automatically. Set these environment variables before starting Julia to override it:

- `LEGATE_SHOW_CONFIG` : When set to 1, the Legate config is printed to stdout
- `LEGATE_AUTO_CONFIG`: When set to 1, Legate will automatically choose the hardware configuration
- `LEGATE_CONFIG`: A string representing the hardware configuration to set

Set `export LEGATE_SHOW_CONFIG=1` to print the hardware configuration when Legate starts.

To configure hardware manually, set `export LEGATE_AUTO_CONFIG=0`, then set
`LEGATE_CONFIG`, for example `export LEGATE_CONFIG="--gpus 1 --cpus 10"`.
Keep the default memory configuration and set the `--gpus` and `--cpus` flags
to limit the resources Legate reserves. See the
[NVIDIA Legate documentation](https://docs.nvidia.com/legate/latest/usage.html#resource-allocation)
for more options.

The same `LEGATE_CONFIG` string can carry logging / profiling flags (for example `--logging legate=debug --log-to-file`). Those are covered under [Debugging](../debugging.md#trace-legate-runtime-work), including how they pair with [CNPreferences](../api_preferences.md#task-scope-names) task-scope naming.

The benchmark harness (`benchmark/run.jl` via `run_benchmark.sh`) sets `LEGATE_CONFIG` from `--gpus` / `--cpus` before each worker starts. See the [benchmarking repository](https://github.com/JuliaLegate/benchmarking).
