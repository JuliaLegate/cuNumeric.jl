# Distributed linear algebra acceptance

Build the in-tree wrapper in developer mode before running these scripts; see
[Developer Mode](../../docs/src/developer_mode.md). These scripts require Linux,
Julia, Bash, GNU `timeout`, and the usual Legate GPU/network configuration.
They do not install packages, change preferences, or start a scheduler allocation.

From the repository root, in an allocation with the requested resources:

```bash
bash scripts/linalg/run.sh cpu
bash scripts/linalg/run.sh one-gpu
CUNUMERIC_LINALG_GPUS_PER_NODE=4 bash scripts/linalg/run.sh multi-gpu
```

For multiple nodes, supply your cluster's working Legate launcher after `--`.
For example, in a Slurm allocation with two nodes and four GPUs per node:

```bash
CUNUMERIC_LINALG_GPUS_PER_NODE=4 CUNUMERIC_LINALG_EXPECT_GPUS=8 \
  bash scripts/linalg/run.sh multi-node -- \
  srun --nodes=2 --ntasks=2 --ntasks-per-node=1
```

Adapt the launcher to the cluster's supported Realm network and GPU binding.
The example runs one Julia/Legate process per node, with four GPUs visible to
each process. A working single-node run does not establish multi-node support.
The script checks the global GPU count reported by Legate against the expected
count; profile inspection must additionally establish participation from each node.

## What runs

- Production selection-policy tests around every cutoff.
- Small real/complex solve, Cholesky, and square/tall/wide QR checks, including
  uneven partitions, empty reduced-output partitions, and input preservation.
- Small forced MP launches whenever multiple GPUs are active, using private
  tile arguments rather than changing module constants.
- Tiled Cholesky, including its complex conjugate-transpose updates.
- Singular solve and non-positive-definite Cholesky, each in a separate process
  under a timeout. Only the expected numerical error counts as success; an
  unrelated exception, crash, or timeout fails the run.

The ordinary package test suite continues to cover promotion and batched
regressions. Run it as well after rebuilding the wrapper:

```bash
LEGATE_CONFIG='--cpus 2 --gpus 0' julia --project -e 'using Pkg; Pkg.test()'
```

To additionally exercise the public APIs at the production MP cutoffs:

```bash
CUNUMERIC_LINALG_PRODUCTION=1 CUNUMERIC_LINALG_GPUS_PER_NODE=4 \
  bash scripts/linalg/run.sh multi-gpu
```

This adds Float64 identity smoke tests, including vector RHS through `A \ b`
and an 8192×8192 Cholesky. Provision enough system and GPU memory for inputs,
outputs, redistribution buffers, and solver workspaces. Cholesky samples columns
at that size; the small tests perform complete nontrivial reconstructions.

## Configuration and evidence

The defaults use two CPU processors and `--profile`. Set `LEGATE_CONFIG` to
provide memory, networking, or other resource flags; include `--profile` to keep
timeline evidence. The runner appends a distinct `--logdir` for each process
launch, so omit that flag from the supplied configuration.

Additional environment variables:

| Variable | Purpose |
| --- | --- |
| `CUNUMERIC_LINALG_JULIA` | Julia executable; defaults to `julia` |
| `CUNUMERIC_LINALG_TIMEOUT` | Timeout per launch in seconds; defaults to 1800 |
| `CUNUMERIC_LINALG_LOGDIR` | Output root; defaults to `linalg-results/<mode>-<timestamp>` |
| `CUNUMERIC_LINALG_PRODUCTION` | `1` enables production-cutoff smoke tests |

Keep the logs containing Julia/library versions, resource configuration, GPU
count, residuals, and test summaries. Process the acceptance profiles with
`legate_prof`. Verify that `MP_SOLVE`, `MP_POTRF`, and `MP_QR` tasks run on the
requested GPUs across every node, rather than inferring distribution from a
passing residual alone. Failure-run profiles are stored separately.

Hardware numerical tests and multi-node acceptance have not been run as part of
this implementation. Report a failure log or timeout as a validation failure;
do not interpret it as a successful fallback or reuse that runtime afterward.
