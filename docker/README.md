# Containers

Build from the repository root:

```bash
docker build -f docker/Dockerfile -t cunumeric:local .
```

The benchmark image and its opt-in CI now live in
[JuliaLegate/benchmarking](https://github.com/JuliaLegate/benchmarking).
Fetch it into `benchmark/` and build without rebuilding the base:

```bash
./scripts/get_benchmarks.sh
cd benchmark
docker build -f docker/Dockerfile \
  --build-arg BASE_IMAGE=cunumeric:local \
  --build-arg BENCHMARK_REF="$(git branch --show-current)" \
  --build-arg BENCHMARK_COMMIT="$(git rev-parse HEAD)" \
  -t cunumeric:benchmark .
```

See [benchmark/docker/README.md](https://github.com/JuliaLegate/benchmarking/blob/main/docker/README.md) for details.

Run the smoke suite with NVIDIA Container Toolkit enabled:

```bash
docker run --rm --gpus=all \
  cunumeric:benchmark \
  julia --project=. run.jl --config=configs/single_gpu/smoke.toml
```
