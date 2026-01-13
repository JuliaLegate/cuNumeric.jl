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

#############################################
# LOTS OF THIS LOGIC EXISTS IN LEGATE.JL TOO
# DEFINITELY DUPLICATED CODE
#############################################

function check_jll(m::Module)
    if !m.is_available()
        m_host_cuda = cupynumeric_jll.host_platform["cuda"]

        if (m_host_cuda == "none")
            error(
                "$(string(m)) installed but not available on this platform.\n $(string(cupynumeric_jll.host_platform))",
            )
        end

        v_host_cuda = VersionNumber(m_host_cuda)
        valid_cuda_version = Legate.MIN_CUDA_VERSION <= v_host_cuda <= Legate.MAX_CUDA_VERSION
        if !valid_cuda_version
            error(
                "$(string(m)) installed but not available on this platform. Host CUDA ver: $(v_host_cuda) not in range supported by $(string(m)): $(MIN_CUDA_VERSION)-$(MAX_CUDA_VERSION).",
            )
        else
            error("$(string(m)) installed but not available on this platform. Unknown reason.")
        end
    end
end

function find_paths(
    mode::String;
    cupynumeric_jll_module::Union{Module,Nothing}=nothing,
    cupynumeric_jll_wrapper_module::Union{Module,Nothing}=nothing,
)
    libcupynumeric_path, libcupynumeric_wrapper_path = cuNumeric._find_paths(
        CNPreferences.to_mode(mode), cupynumeric_jll_module, cupynumeric_jll_wrapper_module
    )
    set_preferences!(CNPreferences, "CUPYNUMERIC_LIBDIR" => libcupynumeric_path; force=true)
    set_preferences!(
        CNPreferences, "CUPYNUMERIC_WRAPPER_LIBDIR" => libcupynumeric_wrapper_path; force=true
    )
end

function _find_paths(
    mode::CNPreferences.JLL,
    cupynumeric_jll_module::Module,
    cupynumeric_jll_wrapper_module::Module,
)
    check_jll(cupynumeric_jll_module)
    check_jll(cupynumeric_jll_wrapper_module)
    legate_lib_dir = joinpath(cupynumeric_jll_module.artifact_dir, "lib")
    legate_wrapper_libdir = joinpath(cupynumeric_jll_wrapper_module.artifact_dir, "lib")
    return legate_lib_dir, legate_wrapper_libdir
end

function _find_paths(
    mode::CNPreferences.Developer,
    cupynumeric_jll_module::Module,
    cupynumeric_jll_wrapper_module::Nothing,
)
    cupynumeric_path = ""
    use_cupynumeric_jll = load_preference(CNPreferences, "cupynumeric_use_jll", true)

    if use_cupynumeric_jll == false
        cupynumeric_path = load_preference(CNPreferences, "cupynumeric_path", nothing)
        check_cupynumeric_install(cupynumeric_path)
    else
        check_jll(cupynumeric_jll_module)
        cupynumeric_path = cupynumeric_jll.artifact_dir
    end

    pkg_root = abspath(joinpath(@__DIR__, "../", "../"))
    wrapper_lib = joinpath(pkg_root, "lib", "cunumeric_jl_wrapper", "build", "lib")

    return joinpath(cupynumeric_path, "lib"), wrapper_lib
end

function _find_paths(
    mode::CNPreferences.Conda,
    cupynumeric_jll_module::Nothing,
    cupynumeric_jll_wrapper_module::Module,
)
    @warn "mode = conda may break. We are using a subset of libraries from conda."

    conda_env = load_preference(CNPreferences, "legate_conda_env", nothing)
    isnothing(conda_env) && error(
        "legate_conda_env preference must be set in LocalPreferences.toml when using conda mode"
    )

    check_legate_install(conda_env)
    legate_path = conda_env
    check_jll(cupynumeric_jll_wrapper_module)
    legate_wrapper_lib = joinpath(cupynumeric_jll_wrapper_module.artifact_dir, "lib")

    return joinpath(legate_path, "lib"), legate_wrapper_lib
end

# MPI, NCCL etc are found by Legate.find_dependency_paths
const DEPS_MAP = Dict(
    "CUTENSOR" => "libcutensor",
    "BLAS" => "libopenblas",
    "TBLIS" => "libtblis",
)
function find_dependency_paths(::Type{CNPreferences.JLL})
    results = Dict{String,String}()

    paths_to_search = copy(cupynumeric_jll.LIBPATH_list)

    for (name, lib) in DEPS_MAP
        results[name] = dirname(Libdl.find_library(lib, paths_to_search))
    end
    return results
end

find_dependency_paths(::Type{CNPreferences.Developer}) = Dict{String,String}()
find_dependency_paths(::Type{CNPreferences.Conda}) = Dict{String,String}()
