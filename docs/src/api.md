
# Public API

User facing functions supported by cuNumeric.jl

```@contents
Pages = ["api.md"]
Depth = 2:2
```

```@autodocs
Modules = [cuNumeric]
Pages = ["cuNumeric.jl", "memory.jl", "util.jl", "ndarray/ndarray.jl", "ndarray/unary.jl", "ndarray/binary.jl"]
```

# CUDA.jl Tasking
This section will detail how to use custom CUDA.jl kernels with the Legate runtime.

```@autodocs
Modules = [cuNumeric]
Pages = ["cuda.jl"]
```

# CNPreferences

This section details how to set custom build configuration options. To see more details visit our install guide [here](./install.md).

```@autodocs
Modules = [CNPreferences]
Pages = ["CNPreferences.jl"]
```

# Internal API

```@autodocs
Modules = [cuNumeric]
Pages = ["ndarray/detail/ndarray.jl"]
```