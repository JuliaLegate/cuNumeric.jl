
# Public API

User facing functions supported by cuNumeric.jl

```@contents
Pages = ["api.md"]
Depth = 2:2
```

```@autodocs
Modules = [cuNumeric]
Pages = ["cuNumeric.jl", "cuda.jl", "memory.jl", "util.jl", "ndarray/ndarray.jl", "ndarray/unary.jl", "ndarray/binary.jl"]
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