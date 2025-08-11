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
    githash_path = joinpath(@__DIR__, "../", ".githash")
    return isfile(githash_path) ? readchomp(githash_path) : "unknown"
end

function version_config_setup()
    project_file = joinpath(@__DIR__, "../", "Project.toml")
    project = TOML.parsefile(project_file)

    name = get(project, "name", "unknown")
    version = get(project, "version", "unknown")
    uuid = get(project, "uuid", "unknown")
    compiler = get_cxx_version(libpath)

    julia_ver = VERSION
    hostname = gethostname()
    git_hash = read_githash()

    liblegate = Legate.get_install_liblegate()
    libnccl = Legate.get_install_libnccl()
    libmpi = Legate.get_install_libmpi()
    libhdf5 = Legate.get_install_libhdf5()
    libcuda = Legate.get_install_libcuda()
    libcudart = Legate.get_install_libcudart()
    liblegatewrapper = Legate.LEGATE_WRAPPER_LIB

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
    CUDA Driver:      $libcuda
    CUDA Runtime:     $libcudart

    Library Paths:
      Legate:         $liblegate
      cuPyNumeric:    $libcupynumeric
      BLAS:           $libblas
      TBLIS:          $libtblis
      CUTENSOR:       $libcutensor
      NCCL:           $libnccl
      MPI:            $libmpi
      HDF5:           $libhdf5

    Wrappers:
      cuNumeric       $libcunumericwrapper
      Legate          $liblegatewrapper
    ───────────────────────────────────────────────
    """
    return str
end
