# Common Errors

## OOM on Startup
If you have other processes using GPU RAM (e.g. another instance of cuNumeric.jl) then cuNumeric.jl will fail to start and will segfault. The first symbol is typically something like `_ZN5Realm4CudaL22allocate_device_memoryEPNS0_3GPUEm`. You can fix this by killing the other jobs or modifying the amount of GPU RAM requested in `LEGATE_CONFIG`. See the [performance](./perf.md) documentation for examples on how to set the `LEGATE_CONFIG` environment variable.
