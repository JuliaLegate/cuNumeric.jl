# we need nvcc as we are compiling during the build of this docker container
# CUDA_Runtime_jll does not contain nvcc 
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
# once we are fully dependent on jlls, we can do the following
ARG JULIA_VERSION=1.10
# FROM julia:${JULIA_VERSION}

# using bash
SHELL ["/bin/bash", "-c"] 
ENV DEBIAN_FRONTEND=noninteractive

# develop mode enables us to build wrappers from source. it installs libcxx wrap into deps of Legate.jl
# this will enable both Legate.jl and/or cuNumeric.jl to build its wrapper from source within this container
ENV LEGATE_DEVELOP_MODE=1
# force turn off legate auto config for precompilation.
ENV LEGATE_AUTO_CONFIG=0

# much of the CUDA.jl setup is from Tim Besard
# CUDA.jl Dockerfile https://github.com/JuliaGPU/CUDA.jl/blob/master/Dockerfile
# Thank you Tim for the reccomendation. 

ARG JULIA_CPU_TARGET=native
ENV JULIA_CPU_TARGET=${JULIA_CPU_TARGET}

# necessary - we are building in this container until we get the jlls for cunumeric
ENV JULIA_NUM_THREADS=auto

ARG CUDA_VERSION=12.4
ARG CUNUMERIC_VERSION=25.05.00

ARG PACKAGE_SPEC_CUDA=CUDA
ARG PACKAGE_SPEC_CUNUMERIC=CUNUMERIC

LABEL org.opencontainers.image.authors="David Krasowska <krasow@u.northwestern.edu>, Ethan Meitz <emeitz@andrew.cmu.edu>" \
      org.opencontainers.image.description="A cuNumeric.jl container with CUDA ${CUDA_VERSION}, Julia ${JULIA_VERSION}, and cuNumeric ${CUNUMERIC_VERSION}" \
      org.opencontainers.image.title="cuNumeric.jl" \
    #   org.opencontainers.image.url="https://juliagpu.org/cuda/" \
      org.opencontainers.image.source="https://github.com/JuliaLegate/cuNumeric.jl" \
      org.opencontainers.image.licenses="MIT"


# system-wide packages
RUN apt-get update && apt-get install -y \
    wget curl git clang libhwloc15 build-essential vim \
    libssl-dev software-properties-common \
    ca-certificates python3-pip libhwloc-dev && \
    rm -rf /var/lib/apt/lists/*

# We need CMake to install cuPyNumeric and the cuNumeric.jl wrapper
# In future releases, we hope to have the jlls available. 
RUN mkdir -p /usr/local && \
    cd $WORKSPACE && \
    wget https://github.com/Kitware/CMake/releases/download/v3.30.7/cmake-3.30.7-linux-x86_64.sh --no-check-certificate && \
    sh cmake-3.30.7-linux-x86_64.sh --skip-license --prefix=/usr/local

# requires Julia install due to not using an official Julia container
RUN curl -fsSL https://install.julialang.org | bash -s -- --default-channel 1.10 --yes \
    --path /usr/local/.juliaup

ENV JULIA_DEPOT_PATH=/usr/local/share/julia:

ENV PATH="/usr/local/cuda-12.4/bin:/usr/local/.juliaup/bin:/usr/local/bin:$PATH"
# Install Julia packages and extract artifact paths
RUN echo "export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda-12.4/lib64:\$LD_LIBRARY_PATH" > /etc/profile.d/custom_ld_library_path.sh

RUN chmod +x /etc/profile.d/custom_ld_library_path.sh
RUN cat /etc/profile.d/custom_ld_library_path.sh


# pre-install the CUDA toolkit from an artifact. we do this separately from CUDA.jl so that
# this layer can be cached independently. it also avoids double precompilation of CUDA.jl in
# order to call `CUDA.set_runtime_version!`.
RUN julia -e '#= make bundled depot non-writable (JuliaLang/Pkg.jl#4120) =# \
              bundled_depot = last(DEPOT_PATH); \
              run(`find $bundled_depot/compiled -type f -writable -exec chmod -w \{\} \;`); \
              #= configure the preference =# \
              env = "/usr/local/share/julia/environments/v$(VERSION.major).$(VERSION.minor)"; \
              mkpath(env); \
              write("$env/LocalPreferences.toml", \
                    "[CUDA_Runtime_jll]\nversion = \"'${CUDA_VERSION}'\""); \
              \
              #= install the JLL =# \
              using Pkg; \
              Pkg.add("CUDA_Runtime_jll"); \
              #= revert bundled depot changes =# \
              run(`find $bundled_depot/compiled -type f -writable -exec chmod +w \{\} \;`)' && \
    #= demote the JLL to an [extras] dep =# \
    find /usr/local/share/julia/environments -name Project.toml -exec sed -i 's/deps/extras/' {} + && \
    #= remove nondeterminisms =# \
    cd /usr/local/share/julia && \
    rm -rf compiled registries scratchspaces logs && \
    find -exec touch -h -d "@0" {} + && \
    touch -h -d "@0" /usr/local/share

# install CUDA.jl itself
RUN julia -e 'using Pkg; pkg"add '${PACKAGE_SPEC_CUDA}'"; \
              using CUDA; CUDA.precompile_runtime()'

# currently, Legate.jl and cuNumeric.jl will fail if the following ENVs are not specified 
# OR if the jll isn't available
# JULIA_NCCL_PATH 
# JULIA_CUTENSOR_PATH
# JULIA_CUDA_PATH

# the reason we do the above is to make the jlls for CUTENSOR, NCCL, and Legate available
# we need is_available() on the jlls to return a success so we can properly determine the
# proper .so file. This requires selecting the CUDA version upfront so this container only 
# supports the version listed in CUDA_VERSION

# TODO [BUG]: Legate.jl throw an error when trying to get the NCCL_jll.artifact_dir, CUTENSOR_jll.artifcat_dir and legate_jll.artifact_dir. 
#       We shouldn't have to do this. This is very hacky. I'm not sure why yet.
RUN julia -e 'using Pkg; Pkg.add("NCCL_jll")'
RUN julia -e 'using NCCL_jll; println(joinpath(NCCL_jll.artifact_dir, "lib"))' > /usr/local/NCCL_PATH.txt
RUN JULIA_NCCL_PATH=$(cat /usr/local/NCCL_PATH.txt) && \
    echo "export JULIA_NCCL_PATH=$JULIA_NCCL_PATH" >> /etc/.env

RUN julia -e 'using Pkg; Pkg.add("CUTENSOR_jll")'
RUN julia -e 'using CUTENSOR_jll; println(joinpath(CUTENSOR_jll.artifact_dir, "lib"))' > /usr/local/CUTENSOR_PATH.txt
RUN JULIA_CUTENSOR_PATH=$(cat /usr/local/CUTENSOR_PATH.txt) && \
    echo "export JULIA_CUTENSOR_PATH=$JULIA_CUTENSOR_PATH" >> /etc/.env

RUN julia -e 'using Pkg; Pkg.add("legate_jll");'
RUN julia -e 'using Pkg; Pkg.add("CUDA_Driver_jll");'
RUN julia -e 'using Pkg; Pkg.add("Libdl");'

RUN julia -e 'using Libdl; using CUDA_Driver_jll; \
              Libdl.dlopen(joinpath(CUDA_Driver_jll.artifact_dir, "lib", "libcuda.so"), Libdl.RTLD_GLOBAL | Libdl.RTLD_NOW); \
              using legate_jll; \
              println(joinpath(legate_jll.artifact_dir))' > /usr/local/LEGATE_PATH.txt

ENV LEGATE_CUSTOM_INSTALL=1
RUN LEGATE_CUSTOM_INSTALL_LOCATION=$(cat /usr/local/LEGATE_PATH.txt) && \
    echo "export LEGATE_CUSTOM_INSTALL_LOCATION=$LEGATE_CUSTOM_INSTALL_LOCATION" >> /etc/.env

RUN echo "Install Legate and cuNumeric.jl"
# Install Legate.jl and cuNumeric.jl
RUN source /etc/.env && julia -e 'using Pkg; Pkg.add(url = "https://github.com/JuliaLegate/Legate.jl", rev = "fixing-build")'
RUN source /etc/.env && julia -e 'using Pkg; Pkg.add(url = "https://github.com/JuliaLegate/cuNumeric.jl", rev = "main")'
RUN source /etc/.env && julia -e 'using Pkg; Pkg.resolve()'

RUN #= remove useless stuff =# \
    cd /usr/local/share/julia && \
    rm -rf registries scratchspaces logs

# user environment

# we hard-code the primary depot regardless of the actual user, i.e., we do not let it
# default to `$HOME/.julia`. this is for compatibility with `docker run --user`, in which
# case there might not be a (writable) home directory.

RUN mkdir -m 0777 /depot
# we add the user environment from a start-up script
# so that the user can mount `/depot` for persistency
COPY <<EOF /usr/local/share/julia/config/startup.jl
if !isdir("/deopt/environments/v$(VERSION.major).$(VERSION.minor)")
    if isinteractive() && Base.JLOptions().quiet == 0
        println("""Welcome to this cuNumeric.jl container!

                   Since this is the first time you're running this container,
                   we'll set up a user depot for you at `/depot`. For persistency,
                   you can mount a volume at this location.

                   The cuNumeric.jl package is pre-installed, and ready to be imported.
                   Remember that you need to invoke Docker with e.g. `--gpus=all`
                   to access the GPU.""")
    end
    mkpath("/deopt/environments")
    cp("/usr/local/share/julia/environments/v$(VERSION.major).$(VERSION.minor)",
       "/deopt/environments/v$(VERSION.major).$(VERSION.minor)")
end
pushfirst!(DEPOT_PATH, "/deopt")
EOF

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV LEGATE_AUTO_CONFIG=1 

ENTRYPOINT source /etc/profile.d/custom_ld_library_path.sh && exec /bin/bash
WORKDIR /workspace