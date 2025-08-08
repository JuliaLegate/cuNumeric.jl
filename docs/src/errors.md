# Common Errors
### [1] ERROR: LoadError: JULIA_LEGATE_XXXX_PATH not found via environment or JLL.
This can occur for several reasons; however, this means the JLL is not available.
For the library that failed, you can overwrite an ENV to use a custom install.
```bash
export JULIA_LEGATE_XXXX_PATH="/path/to/library/failing"
```

However, if you want to solve the JLL being available- you need the cuda driver `libcuda.so` on your path and cuda runtime `libcudart.so` on your path. You can use JLLs to achieve this:

```bash
echo "LD_LIBRARY_PATH=$(julia -e 'using Pkg; \
    Pkg.add(name = "CUDA_Driver_jll", version = "0.12.1"); \
    using CUDA_Driver_jll; \
    print(joinpath(CUDA_Driver_jll.artifact_dir, "lib"))' \
):$LD_LIBRARY_PATH" 
```

Note: You may use a different compatible driver version, but ensure it works with our supported CUDA toolkit/runtime versions (12.2 – 12.9). CUDA runtime 13.0 is untested and will likely break this package.