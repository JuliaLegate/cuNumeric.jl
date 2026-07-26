# Hardware Configuration

There is no programmatic way to set the hardware configuration used by CuPyNumeric (as of 26.01). By default, the hardware configuration is set automatically by Legate. This configuration can be manipulated through the following environment variables:

- `LEGATE_SHOW_CONFIG` : When set to 1, the Legate config is printed to stdout
- `LEGATE_AUTO_CONFIG`: When set to 1, Legate will automatically choose the hardware configuration
- `LEGATE_CONFIG`: A string representing the hardware configuration to set

These variables must be set before launching the Julia instance running cuNumeric.jl. We recommend setting `export LEGATE_SHOW_CONFIG=1` so that the hardware configuration will be printed when Legate starts. This output is automatically captured and relayed to the user.

To manually set the hardware configuration, `export LEGATE_AUTO_CONFIG=0`, and then define your own config with something like `export LEGATE_CONFIG="--gpus 1 --cpus 10"`. We recommend using the default memory configuration for your machine and only setting the `gpus`, `cpus`. More details about the Legate configuration can be found in the [NVIDIA Legate documentation](https://docs.nvidia.com/legate/latest/usage.html#resource-allocation).

The same `LEGATE_CONFIG` string can carry logging / profiling flags (for example `--logging legate=debug --log-to-file`). Those are covered under [Debugging](../debugging.md#inspect-legate-with-logging-and-task-scope-names), including how they pair with [CNPreferences](../api_preferences.md#task-scope-names) task-scope naming.

The benchmark harness (`benchmark/run.jl` via `run_benchmark.sh`) sets `LEGATE_CONFIG` from `--gpus` / `--cpus` before each worker starts. See [How to Benchmark](../benchmarks/howto.md).
