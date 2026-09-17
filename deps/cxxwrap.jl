# A version marker alone cannot validate a build-tree CMake export: its headers
# can live in a checkout that has since been deleted or moved.
function cxxwrap_usable(override_dir; log_dir)
    mktempdir() do build_dir
        probe = joinpath(@__DIR__, "cxxwrap_probe")
        julia = joinpath(Sys.BINDIR, Base.julia_exename())
        cmd = `cmake -S $probe -B $build_dir -DJlCxx_DIR=$override_dir -DJulia_EXECUTABLE=$julia`
        open(joinpath(log_dir, "libcxxwrap_check.log"), "w") do io
            return success(pipeline(cmd; stdout=io, stderr=io))
        end
    end
end

function ensure_cxxwrap(repo_root, package_version; log_dir)
    override_dir = joinpath(DEPOT_PATH[1], "dev", "libcxxwrap_julia_jll", "override")
    version_path = joinpath(override_dir, "LEGATE_INSTALL.txt")
    cached = isfile(version_path) ? tryparse(VersionNumber, strip(read(version_path, String))) : nothing
    if !isnothing(cached) && is_supported_version(cached) &&
       cxxwrap_usable(override_dir; log_dir)
        @info "libcxxwrap: Up to date (version $cached)"
        return nothing
    end

    @info "libcxxwrap: Missing, incompatible, or stale build. Rebuilding..."
    # Invalidate before attempting the build, and propagate failures. The shared
    # run_sh helper catches errors, so it cannot establish a successful build.
    rm(version_path; force=true)
    script = joinpath(repo_root, "scripts", "install_cxxwrap.sh")
    cmd = addenv(`bash $script $repo_root`, "JULIA" => joinpath(Sys.BINDIR, Base.julia_exename()))
    open(joinpath(log_dir, "libcxxwrap.log"), "w") do out
        open(joinpath(log_dir, "libcxxwrap.err"), "w") do err
            try
                run(pipeline(cmd; stdout=out, stderr=err))
            catch
                error("libcxxwrap build failed; see $(joinpath(log_dir, "libcxxwrap.err"))")
            end
        end
    end
    cxxwrap_usable(override_dir; log_dir) ||
        error("libcxxwrap build produced an unusable CMake package; see $(joinpath(log_dir, "libcxxwrap_check.log"))")
    write(version_path, string(package_version))
    return nothing
end
