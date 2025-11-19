function is_cupynumeric_installed(cupynumeric_root::String; throw_errors::Bool=false)
    include_dir = joinpath(cupynumeric_root, "include")
    if !isdir(joinpath(include_dir, "cupynumeric"))
        throw_errors &&
            @error "cuNumeric.jl: Cannot find include/cupynumeric in $(cupynumeric_root)"
        return false
    end
    return true
end

function parse_cupynumeric_version(cupynumeric_root)
    version_file = joinpath(cupynumeric_root, "include", "cupynumeric", "version_config.hpp")
    version = nothing
    open(version_file, "r") do f
        data = readlines(f)
        major = parse(Int, split(data[end - 2])[end])
        minor = lpad(split(data[end - 1])[end], 2, '0')
        patch = lpad(split(data[end])[end], 2, '0')
        version = "$(major).$(minor).$(patch)"
    end

    if isnothing(version)
        error("cuNumeric.jl: Failed to parse version from conda environment")
    end

    return version
end

function check_cupynumeric_install(cupynumeric_root)
    cupynumeric_installed = is_cupynumeric_installed(cupynumeric_root)
    if !cupynumeric_installed
        error("cuNumeric.jl: Build halted: cupynumeric not found in $cupynumeric_root")
    end
    installed_version = parse_cupynumeric_version(cupynumeric_root)
    if installed_version ∉ SUPPORTED_CUPYNUMERIC_VERSIONS
        error(
            "cuNumeric.jl: Build halted: $(cupynumeric_root) detected unsupported version $(installed_version)"
        )
    end
    @info "cuNumeric.jl: Found a valid install in: $(cupynumeric_root)"
    return true
end

function get_library_root(jll_module, env_var::String)
    if haskey(ENV, env_var)
        return get(ENV, env_var, "0")
    elseif jll_module.is_available()
        return joinpath(jll_module.artifact_dir, "lib")
    else
        error("$env_var not found via environment or JLL.")
    end
end

function find_preferences()
    pkg_root = abspath(joinpath(@__DIR__, "../", "../"))

    blas_lib = get_library_root(OpenBLAS32_jll, "JULIA_OPENBLAS_PATH")
    if HAS_CUDA
        cutensor_lib = get_library_root(CUTENSOR_jll, "JULIA_CUTENSOR_PATH")
    end

    cupynumeric_path = cupynumeric_jll.artifact_dir

    mode = load_preference(CNPreferences, "mode", CNPreferences.MODE_JLL)

    # if developer mode
    if mode == CNPreferences.MODE_JLL
        cunumeric_wrapper_lib = joinpath(cunumeric_jl_wrapper_jll.artifact_dir, "lib")
    elseif mode == CNPreferences.MODE_DEVELOPER
        use_cupynumeric_jll = load_preference(
            CNPreferences, "use_cupynumeric_jll", CNPreferences.DEVEL_DEFAULT_JLL_CONFIG
        )
        if use_cupynumeric_jll == false
            cupynumeric_path = load_preference(
                CNPreferences, "cupynumeric_path", CNPreferences.DEVEL_DEFAULT_CUPYNUMERIC_PATH
            )
            check_cupynumeric_install(cupynumeric_path)
        end
        cunumeric_wrapper_lib = joinpath(pkg_root, "deps", "cunumeric_jl_wrapper", "lib")
        # if conda
    elseif mode == CNPreferences.MODE_CONDA
        @warn "mode = conda may break. We are using a subset of libraries from conda."
        conda_env = load_preference(CNPreferences, "conda_env", nothing)
        check_cupynumeric_install(conda_env)
        cupynumeric_path = conda_env
        cutensor_lib = joinpath(conda_env, "lib")
    end

    cupynumeric_lib = joinpath(cupynumeric_path, "lib")
    if haskey(ENV, "JULIA_TBLIS_PATH")
        tblis_lib = get(ENV, "JULIA_TBLIS_PATH", "0")
    else
        tblis_lib = cupynumeric_lib # cupynumeric libpath will by default contain tblis
    end

    if HAS_CUDA
        set_preferences!(CNPreferences, "CUTENSOR_LIB" => cutensor_lib; force=true)
    end

    set_preferences!(CNPreferences, "BLAS_LIB" => blas_lib; force=true)
    set_preferences!(CNPreferences, "TBLIS_LIB" => tblis_lib; force=true)
    set_preferences!(CNPreferences, "CUPYNUMERIC_LIB" => cupynumeric_lib; force=true)
    set_preferences!(CNPreferences, "CUNUMERIC_WRAPPER_LIB" => cunumeric_wrapper_lib; force=true)
end
