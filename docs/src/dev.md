# Developing cuNumeric.jl

To contribute to cuNumeric.jl, we recommend cloning the repository and adding it to one of your existing environments with `Pkg.develop`.
```bash
git clone https://github.com/JuliaLegate/cuNumeric.jl.git
julia --project=. -e 'using Pkg; Pkg.develop(path = "cuNumeric.jl/lib/CNPreferences")'
julia --project=. -e 'using Pkg; Pkg.develop(path = "cuNumeric.jl")'
julia --project=. -e 'using CNPreferences; CNPreferences.use_developer_mode()'
julia --project=. -e 'using Pkg; Pkg.build()'
```
