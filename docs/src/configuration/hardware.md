# Hardware Configuration

Legate manages the processors and memory used by cuNumeric.jl. By default it
detects the available hardware and reserves most of its resources. On a shared
machine, or when running several Julia processes, explicitly limit each
process's allocation to avoid competing for the same CPUs, GPUs, and memory.

Set environment variables **before starting Julia**. The runtime reads its
configuration once during initialization. Changing `ENV` after `using cuNumeric`
does not reconfigure it. Restart Julia to apply changes. The shell examples below
use Bash on Linux.

## Automatic and manual configuration

| Environment variable | Meaning |
|:---|:---|
| `LEGATE_SHOW_CONFIG=1` | Print the resolved configuration at startup. |
| `LEGATE_AUTO_CONFIG=1` | Detect defaults for unspecified resources (the default). Explicit `LEGATE_CONFIG` options override those defaults. |
| `LEGATE_AUTO_CONFIG=0` | Disable detection and use minimal defaults: one CPU task processor, no GPUs, and a small system-memory allocation. Specify the resources your program needs. |
| `LEGATE_CONFIG` | Space-separated runtime options, such as `"--gpus 1 --fbmem 4096"`. |

To select one GPU while keeping automatic memory sizing:

```bash
export LEGATE_AUTO_CONFIG=1
export LEGATE_CONFIG="--gpus 1 --cpus 1"
julia --project myscript.jl
```

Automatic memory sizing can still reserve most of the available memory. For a
shared machine, set memory limits as well. For example, this fully manual
configuration requests one GPU, one CPU task processor, 4 GiB of system memory,
and 4 GiB of device memory **per GPU** (`--fbmem`):

```bash
export LEGATE_AUTO_CONFIG=0
export LEGATE_CONFIG="--gpus 1 --cpus 1 --sysmem 4096 --fbmem 4096"
julia --project myscript.jl
```

Choose memory sizes that fit your workload and leave room for Julia, CUDA,
runtime overhead, and other processes. These settings reserve runtime memory
pools. They do not cap the entire process's memory usage.

## Inspect the configuration

After starting Julia with your chosen configuration, print a summary:

```julia
using cuNumeric
cuNumeric.versioninfo()
```

This includes hardware support, the auto-config setting, the active GPU count,
and build details. When `LEGATE_AUTO_CONFIG=0`, it also prints the
`LEGATE_CONFIG` string. With auto-configuration enabled, that field reads `auto`.

To see the full resolved settings, including automatically selected memory sizes,
enable configuration output when starting Julia:

```bash
LEGATE_SHOW_CONFIG=1 julia --project -e 'using cuNumeric'
```

To list the options supported by the installed runtime:

```bash
LEGATE_CONFIG="--help" julia --project -e 'using cuNumeric'
```

The help command prints the runtime options and exits. No separate Python
environment or `legate` launcher is needed for these Julia commands.

## Processor and memory options

Processor counts apply to each Legate process (rank). Memory values below are in
**MiB** (1024 MiB = 1 GiB).

| Option | Resource controlled |
|:---|:---|
| `--gpus N` | Number of visible GPUs to use. `0` disables GPU task execution. |
| `--cpus N` | Number of CPU task processors. This does not set Julia's thread count. |
| `--sysmem M` | System-memory pool per process. |
| `--fbmem M` | Device-memory pool **per GPU**. Two GPUs with `--fbmem 4096` reserve 8 GiB in total. |

For CPU task execution with four task processors and an 8 GiB memory pool:

```bash
LEGATE_AUTO_CONFIG=0 LEGATE_CONFIG="--gpus 0 --cpus 4 --sysmem 8192" julia --project myscript.jl
```

## Select GPUs and share a machine

`--gpus` is a count, not a list of device IDs. Use `CUDA_VISIBLE_DEVICES` to
select devices before starting Julia. For example, on a machine with GPUs 0–3:

```bash
CUDA_VISIBLE_DEVICES=0,2 LEGATE_AUTO_CONFIG=0 \
  LEGATE_CONFIG="--gpus 2 --cpus 1 --sysmem 4096 --fbmem 4096" \
  julia --project myscript.jl
```

The same `--fbmem` value applies to each selected GPU. If their capacities differ,
choose a value that fits the smallest selected device, or exclude it. On a
scheduler-managed cluster, respect the GPU visibility and CPU allocation assigned
to your job.

For concurrent jobs, give each process its own devices where possible and budget
host memory across all processes. If startup fails while reserving GPU memory,
check device usage with `nvidia-smi`, choose an available GPU, or lower `--fbmem`.
If host-memory reservation fails, try lowering `--sysmem`.

## Further reference

The options above are documented in the archived Legate 26.06.01
[runtime guide](https://github.com/nv-legate/legate/blob/v26.06.01/docs/legate/source/manual/usage/running.rst)
and [runtime option definitions](https://github.com/nv-legate/legate/blob/v26.06.01/src/cpp/legate/runtime/detail/argument_parsing/parse.cc).
These versioned source references replace links to the retired documentation
site. Use `LEGATE_CONFIG="--help"` for the options available in your installed
version.

The same `LEGATE_CONFIG` string can carry logging / profiling flags (for example `--logging legate=debug --log-to-file`). Those are covered under [Debugging](../debugging.md#Trace-Legate-runtime-work), including how they pair with [CNPreferences](../api_preferences.md#Task-scope-names) task-scope naming.
