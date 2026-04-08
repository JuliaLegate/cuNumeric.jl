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

@doc"""
    versioninfo()

Prints the cuNumeric build configuration summary, including package
metadata, Julia and compiler version, and paths to core dependencies.
"""
function versioninfo(io::IO=stdout)
    name = string(Base.nameof(@__MODULE__))
    version = string(Base.pkgversion(cuNumeric))
    compiler = get_cxx_version(CUPYNUMERIC_LIB_PATH)

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

    hardware_str = HAS_CUDA ? "CPU + GPU" : "CPU Only"

    legate_auto_config = get(ENV, "LEGATE_AUTO_CONFIG", "1")
    is_auto_config = legate_auto_config != "0" ? true : false
    legate_config = is_auto_config ? "auto" : get(ENV, "LEGATE_CONFIG", "not set")

    str = """
    ───────────────────────────────────────────────
    cuNumeric Build Configuration
    ───────────────────────────────────────────────
    Package Name:       $name
    Version:            $version
    Git Commit:         $git_hash
    Hardware Support:   $hardware_str
    Legate Auto Config: $is_auto_config
    Legate Config:      $legate_config

    Hostname:         $hostname
    Julia Version:    $(VERSION)
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
    println(io, str)
end
