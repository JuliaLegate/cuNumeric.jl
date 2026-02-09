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
    compiler = get_cxx_version(CUPYNUMERIC_LIB_PATH)

    julia_ver = VERSION
    hostname = gethostname()
    git_hash = read_githash()

    liblegate = Legate.LEGATE_LIBDIR
    liblegatewrapper = Legate.LEGATE_WRAPPER_LIBDIR
    other_dirs = Dict()

    cn_mode = CNPreferences.to_mode(CNPreferences.MODE)
    legate_mode = LegatePreferences.to_mode(LegatePreferences.MODE)
    dirs1 = cuNumeric.find_dependency_paths(typeof(cn_mode))
    dirs2 = Legate.find_dependency_paths(typeof(legate_mode))
    other_dirs = merge(dirs1, dirs2)

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
    CUDA Driver:      $(get(other_dirs,"CUDA_DRIVER","unknown"))
    CUDA Runtime:     $(get(other_dirs,"CUDA_RUNTIME","unknown"))

    Library Paths:
      Legate:         $liblegate
      cuPyNumeric:    $(CUPYNUMERIC_LIBDIR)
      BLAS:           $(get(other_dirs,"BLAS","unknown"))
      TBLIS:          $(get(other_dirs,"TBLIS","unknown"))
      CUTENSOR:       $(get(other_dirs,"CUTENSOR","unknown"))
      NCCL:           $(get(other_dirs,"NCCL","unknown"))
      MPI:            $(get(other_dirs,"MPI","unknown"))
      HDF5:           $(get(other_dirs,"HDF5","unknown"))

    Wrappers:
      cuNumeric       $(CUPYNUMERIC_WRAPPER_LIBDIR)
      Legate          $liblegatewrapper
    
    Modes:
      cuNumeric:      $(CNPreferences.MODE)
      Legate:         $(LegatePreferences.MODE)
    ───────────────────────────────────────────────
    """
    return str
end
