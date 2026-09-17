# Containers

Build from the repository root:

```bash
docker build -f docker/Dockerfile -t cunumeric:local .
docker build \
  -f docker/Dockerfile.benchmark \
  --build-arg BASE_IMAGE=cunumeric:local \
  -t cunumeric:benchmark .
```

Run the smoke suite with NVIDIA Container Toolkit enabled:

```bash
docker run --rm --gpus=all \
  cunumeric:benchmark \
  julia --project=. run.jl --config=benchmarks_smoke.toml
```
