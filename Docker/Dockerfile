# we need nvcc as we are compiling during the build of this docker container
# CUDA_Runtime_jll does not contain nvcc 
ARG CUDA_VERSION=12.4
ENV CUDA_VERSION=${CUDA_VERSION}

ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}.0-devel-ubuntu${UBUNTU_VERSION}
# once we are fully dependent on jlls, we can do the following
ARG JULIA_VERSION=1.10
# FROM julia:${JULIA_VERSION}

ARG REF=main
ENV REF=${REF}
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

ARG CUNUMERIC_VERSION=25.05.00
ARG PACKAGE_SPEC_CUDA=CUDA

LABEL org.opencontainers.image.authors="David Krasowska <krasow@u.northwestern.edu>, Ethan Meitz <emeitz@andrew.cmu.edu>" \
      org.opencontainers.image.description="A cuNumeric.jl container with CUDA ${CUDA_VERSION}, Julia ${JULIA_VERSION}, and cuNumeric ${CUNUMERIC_VERSION}" \
      org.opencontainers.image.title="cuNumeric.jl" \
    #   org.opencontainers.image.url="https://juliagpu.org/cuda/" \
      org.opencontainers.image.source="https://github.com/JuliaLegate/cuNumeric.jl" \
      org.opencontainers.image.licenses="MIT"


# # system-wide packages
RUN apt-get update && apt-get install -y \
    wget curl git build-essential && \
    rm -rf /var/lib/apt/lists/*

# We need CMake to install cuPyNumeric and the cuNumeric.jl wrapper
# In future releases, we hope to have the jlls available. 
RUN mkdir -p /usr/local && \
    cd $WORKSPACE && \
    wget https://github.com/Kitware/CMake/releases/download/v3.30.7/cmake-3.30.7-linux-x86_64.sh --no-check-certificate && \
    sh cmake-3.30.7-linux-x86_64.sh --skip-license --prefix=/usr/local

# requires Julia install due to not using an official Julia container
RUN curl -fsSL https://install.julialang.org | bash -s -- --default-channel ${JULIA_VERSION} --yes \
    --path /usr/local/.juliaup

ENV JULIA_DEPOT_PATH=/usr/local/share/julia:

ENV PATH="/usr/local/cuda-${CUDA_VERSION}/bin:/usr/local/.juliaup/bin:/usr/local/bin:$PATH"
# Install Julia packages and extract artifact paths
RUN echo "export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda-${CUDA_VERSION}/lib64:\$LD_LIBRARY_PATH" > /etc/.env

RUN chmod +x /etc/.env
RUN cat /etc/.env

# install CUDA.jl itself
RUN julia --color=yes -e 'using Pkg; Pkg.add("CUDA"); using CUDA; CUDA.set_runtime_version!(VersionNumber(ENV["CUDA_VERSION"]))'
RUN echo "Install Legate and cuNumeric.jl"
# Install Legate.jl and cuNumeric.jl
RUN source /etc/.env && julia -e 'using Pkg; Pkg.add(url = "https://github.com/JuliaLegate/Legate.jl", rev = "doc-test")'
RUN source /etc/.env && julia -e "using Pkg; Pkg.add(url = \"https://github.com/JuliaLegate/cuNumeric.jl\", rev = \"$REF\")"
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
if !isdir("/depot/environments/v$(VERSION.major).$(VERSION.minor)")
    if isinteractive() && Base.JLOptions().quiet == 0
        println("""Welcome to this cuNumeric.jl container!

                   Since this is the first time you're running this container,
                   we'll set up a user depot for you at `/depot`. For persistency,
                   you can mount a volume at this location.

                   The cuNumeric.jl package is pre-installed, and ready to be imported.
                   Remember that you need to invoke Docker with e.g. `--gpus=all`
                   to access the GPU.""")
    end
    mkpath("/depot/environments")
    cp("/usr/local/share/julia/environments/v$(VERSION.major).$(VERSION.minor)",
       "/depot/environments/v$(VERSION.major).$(VERSION.minor)")
end
pushfirst!(DEPOT_PATH, "/depot")
EOF

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV LEGATE_AUTO_CONFIG=1 

ENTRYPOINT source /etc/.env && exec /bin/bash
WORKDIR /workspace