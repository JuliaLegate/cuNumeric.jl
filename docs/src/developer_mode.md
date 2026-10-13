# Developer Mode

`````@raw html
<!--
Developer mode builds the Julia C++ wrapper from the sources in this repo under `lib/cunumeric_jl_wrapper`, instead of downloading a prebuilt `cunumeric_jl_wrapper_jll`. Use it when you change the wrapper, debug the C-API boundary, or point at a custom cuPyNumeric install.
-->
`````

Developer mode builds the Julia C++ wrapper from the sources in this repo under `lib/cunumeric_jl_wrapper`, instead of downloading a prebuilt `cunumeric_jl_wrapper_jll`. Use it when you change the wrapper or debug the C-API boundary.

## When to use it

- You edited C++ or CxxWrap code under `lib/cunumeric_jl_wrapper/` and need a new `.so`.
`````@raw html
<!--
- You are pairing cuNumeric.jl against a cuPyNumeric build that is not the JLL artifact (for example a source tree).
-->
`````

- You want faster edit-and-rebuild cycles on the wrapper without waiting on a JLL release.

Ordinary Julia-only changes under `src/` do **not** require developer mode. Restart or re-`using` as usual.

## Enable developer mode

From an environment that can load `CNPreferences` (the docs project or a development checkout of cuNumeric.jl):

```julia
using CNPreferences
CNPreferences.use_developer_mode()
```

`````@raw html
<!--
- `use_jll=true` (default): still use the `cupynumeric_jll` binary, but compile the wrapper in-tree against it.
- `use_jll=false, path="..."`: compile the wrapper against a cuPyNumeric root you already installed. See [Build Modes](./install.md) for compiler and CMake requirements.
-->
`````

This builds the wrapper against the prebuilt `cupynumeric_jll` binary.

Preferences are written to `LocalPreferences.toml`. **Restart Julia** after changing the mode.

## Rebuild the wrapper after changes

After editing files under `lib/cunumeric_jl_wrapper/` (or after switching into developer mode for the first time):

```julia
using Pkg
Pkg.build("cuNumeric")
```

That runs `deps/build.jl`, which in developer mode:

1. Resolves the prebuilt cuPyNumeric binary
2. Builds or updates the CxxWrap dependencies as needed
3. Compiles `lib/cunumeric_jl_wrapper` via `scripts/build_cpp_wrapper.sh` into `lib/cunumeric_jl_wrapper/build/`
4. Points the package at that local library (artifact override)

Restart Julia to load the rebuilt shared library.

If the build fails, check CMake / g++ (C++20) / CUDA toolkit availability as described on [Build Modes](./install.md). Build logs from the helper scripts are written under `deps/`.

Recreating or moving the repository can leave a libcxxwrap CMake export in the
Julia depot pointing at deleted headers. `Pkg.build("cuNumeric")` checks the
exported header and library paths before reusing that build and rebuilds it when
stale. It also records the Julia version and executable path: changing either
forces a rebuild. Older builds without this Julia marker are rebuilt once.
No manual checkout or depot cleanup is needed. The installer retains your
manifest and developed JLL checkout, recreating only its generated `override/`
directory. Check `deps/libcxxwrap_check.log`, `deps/libcxxwrap.log`, and
`deps/libcxxwrap.err` if this repair fails.

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

`````@raw html
<!--
- [Build Modes](./install.md): JLL, developer, and conda providers
-->
`````

- [Build Modes](./install.md): prebuilt and developer wrapper modes
- [CNPreferences](./api_preferences.md): preference defaults and function reference
- [Debugging](./debugging.md): fusion and lifetime printers while developing
- [Internals](./internals.md): how fusion and `@accelerate` work
