# Developer Mode

Developer mode builds the Julia C++ wrapper from the sources in this repo under `lib/cunumeric_jl_wrapper`, instead of downloading a prebuilt `cunumeric_jl_wrapper_jll`. Use it when you change the wrapper, debug the C-API boundary, or point at a custom cupynumeric install.

## When to use it

- You edited C++ or CxxWrap code under `lib/cunumeric_jl_wrapper/` and need a new `.so`.
- You are pairing cuNumeric.jl against a cupynumeric build that is not the JLL artifact (for example a source tree).
- You want faster iterate-rebuild cycles on the wrapper without waiting on a JLL release.

Ordinary Julia-only changes under `src/` do **not** require developer mode. Restart or re-`using` as usual.

## Enable developer mode

From an environment that can load `CNPreferences` (the docs project or a develop checkout of cuNumeric.jl):

```julia
using CNPreferences
CNPreferences.use_developer_mode(; use_jll=true, path=nothing)
```

- `use_jll=true` (default): still use the `cupynumeric_jll` binary, but compile the wrapper in-tree against it.
- `use_jll=false, path="..."`: compile the wrapper against a cupynumeric root you already installed. See [Build Modes](./install.md) for compiler and CMake requirements.

Preferences are written to `LocalPreferences.toml`. **Restart Julia** after changing the mode.

## Rebuild the wrapper after changes

After editing files under `lib/cunumeric_jl_wrapper/` (or after switching into developer mode for the first time):

```julia
using Pkg
Pkg.build("cuNumeric")
```

That runs `deps/build.jl`, which in developer mode:

1. Resolves cupynumeric (JLL or your `path`)
2. Builds / refreshes the CxxWrap pieces as needed
3. Compiles `lib/cunumeric_jl_wrapper` via `scripts/build_cpp_wrapper.sh` into `lib/cunumeric_jl_wrapper/build/`
4. Points the package at that local library (artifact override)

Then restart Julia (or at least reload cuNumeric) so the new shared library is picked up.

If the build fails, check CMake / g++ (C++20) / CUDA toolkit availability as described on [Build Modes](./install.md). Build logs from the helper scripts are written under `deps/`.

### Typical edit loop

```text
1. Edit lib/cunumeric_jl_wrapper/...
2. Pkg.build("cuNumeric")
3. Restart Julia
4. using cuNumeric
```

## Switch back to JLLs

When you no longer need a local wrapper:

```julia
using CNPreferences
CNPreferences.use_jll_binary()
```

Restart Julia. You do not need `Pkg.build` for pure JLL mode (the build script exits early).

## Related pages

- [Build Modes](./install.md): JLL, developer, and conda providers
- [CNPreferences](./api_preferences.md): preference defaults and function reference
- [Debugging](./debugging.md): fusion and lifetime printers while developing
- [Internals](./internals.md): how fusion and `@analyze_lifetimes` work
