
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

function version_config_setup()
    project_file = joinpath(@__DIR__, "../", "Project.toml")
    project = TOML.parsefile(project_file)

    name = get(project, "name", "unknown")
    version = get(project, "version", "unknown")
    uuid = get(project, "uuid", "unknown")
    compiler = get_cxx_version(libpath)

    julia_ver = VERSION
    hostname = gethostname()

    git_hash = try
        readchomp(`git -C $(dirname(project_file)) rev-parse HEAD`)
    catch
        "unknown"
    end

    liblegate = Legate.get_install_liblegate()
    libnccl = Legate.get_install_libnccl()
    libmpi = Legate.get_install_libmpi()
    libhdf5 = Legate.get_install_libhdf5()

    cutensor_root = CUTENSOR_ROOT
    cupynumeric_root = CUPYNUMERIC_ROOT
    tblis_root = TBLIS_ROOT

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

    Library Paths:
      Legate:         $liblegate
      cuPyNumeric:    $cupynumeric_root
      TBLIS:          $tblis_root
      CUTENSOR:       $cutensor_root
      NCCL:           $libnccl
      MPI:            $libmpi
      HDF5:           $libhdf5
    ───────────────────────────────────────────────
    """
    return str
end
