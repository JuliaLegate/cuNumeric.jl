function update_project(version::String)
    Pkg.compat("cupynumeric_jll", version)
    Pkg.compat("Legate", version)

    path = "Project.toml"
    project = TOML.parsefile(path)
    project["version"] = version

    open(path, "w") do io
        TOML.print(io, project)
    end
end

function get_cxx_version(libpath::AbstractString)
    try
        cmd = `readelf -p .comment $libpath`
        output = read(cmd, String)
        for line in split(output, '\n')
            if occursin("GCC:", line) || occursin("Clang:", line)
                m = match(r"\] *(.*)", line)
                return m === nothing ? strip(line) : strip(m.captures[1])
            end
        end
        return "unknown compiler"
    catch e
        return "error: $(e)"
    end
end

function read_githash()
    githash_path = joinpath(@__DIR__, "../", "../", ".githash")
    return isfile(githash_path) ? readchomp(githash_path) : "unknown"
end

function version_config_setup()
    project_file = joinpath(@__DIR__, "../", "../", "Project.toml")
    project = TOML.parsefile(project_file)

    name = get(project, "name", "unknown")
    version = get(project, "version", "unknown")
    uuid = get(project, "uuid", "unknown")
    compiler = get_cxx_version(libpath)

    julia_ver = VERSION
    hostname = gethostname()
    git_hash = read_githash()

    liblegate = Legate.LEGATE_LIBDIR
    liblegatewrapper = Legate.LEGATE_WRAPPER_LIBDIR
    if Legate.LegatePreferences.MODE == "jll"
        other_dirs = Legate.find_dependency_paths(Legate.JLL())
    else
        other_dirs = Dict(
            "HDF5" => "unknown",
            "MPI" => "unknown",
            "NCCL" => "unknown",
            "CUDA_DRIVER" => "unknown",
            "CUDA_RUNTIME" => "unknown",
        )
    end

    libblas = BLAS_LIB
    libcutensor = CUTENSOR_LIB
    libcupynumeric = CUPYNUMERIC_LIB
    libtblis = TBLIS_LIB
    libcunumericwrapper = CUNUMERIC_WRAPPER_LIB

    str = """
    ───────────────────────────────────────────────
    cuNumeric Build Configuration
    ───────────────────────────────────────────────
    Package Name:     $name
    Version:          $version
    UUID:             $uuid
    Git Commit:       $git_hash

    Hostname:         $hostname
    Julia Version:    $julia_ver
    C++ Compiler:     $compiler
    CUDA Driver:      $(other_dirs["CUDA_DRIVER"])
    CUDA Runtime:     $(other_dirs["CUDA_RUNTIME"])

    Library Paths:
      Legate:         $liblegate
      cuPyNumeric:    $libcupynumeric
      BLAS:           $libblas
      TBLIS:          $libtblis
      CUTENSOR:       $libcutensor
      NCCL:           $(other_dirs["NCCL"])
      MPI:            $(other_dirs["MPI"])
      HDF5:           $(other_dirs["HDF5"])

    Wrappers:
      cuNumeric       $libcunumericwrapper
      Legate          $liblegatewrapper
    ───────────────────────────────────────────────
    """
    return str
end
