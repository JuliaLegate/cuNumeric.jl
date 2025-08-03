##### temporary step for development branch

using Tar
using Downloads
using CodecZlib

function cunumeric_wrapper_jll_local_branch_install()
    url = "https://github.com/krasow/Yggdrasil/releases/download/v25.5.0/cunumeric_jl_wrapper.v25.5.0.x86_64-linux-gnu-cxx11-julia_version+1.10.0-cuda+12.4.tar.gz"
    dest = joinpath(pwd(), basename(url))
    extract_dir = joinpath(pwd(), "extracted_cunumeric")

    # Download only if not already present
    isfile(dest) || Downloads.download(url, dest)

    # Remove old extraction dir if exists
    isdir(extract_dir) && rm(extract_dir; recursive=true)
    mkpath(extract_dir)

    # Decompress and extract
    open(dest) do io
        Tar.extract(GzipDecompressorStream(io), extract_dir)
    end

    return joinpath(extract_dir, "lib")
end
