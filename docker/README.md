# Containers

Build from the repository root:

```bash
docker build -f docker/Dockerfile -t cunumeric:local .
```

The benchmark image and its opt-in CI now live in
[JuliaLegate/benchmarking](https://github.com/JuliaLegate/benchmarking).
Build it from the pinned submodule without rebuilding the base:

```bash
git submodule update --init --recursive
docker build -f benchmark/docker/Dockerfile \
  --build-arg BASE_IMAGE=cunumeric:local \
  -t cunumeric:benchmark benchmark
```

Run the smoke suite with NVIDIA Container Toolkit enabled:

```bash
docker run --rm --gpus=all \
  cunumeric:benchmark \
  julia --project=. run.jl --config=benchmarks_smoke.toml
```
