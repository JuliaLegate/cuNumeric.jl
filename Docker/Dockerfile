FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/home/user
ENV JULIA_NUM_THREADS=auto
ENV LEGATE_DEVELOP_MODE=1

RUN apt-get update && apt-get install -y \
    wget curl git clang libhwloc15 build-essential \
    libssl-dev software-properties-common \
    ca-certificates python3-pip libhwloc-dev && \
    rm -rf /var/lib/apt/lists/*

    # Install Julia
RUN curl -fsSL https://install.julialang.org | bash -s -- --default-channel 1.10 --yes
ENV PATH="$HOME/.juliaup/bin:$PATH"

# Install CMake to ~/.local
RUN mkdir -p $HOME/.local && \
    cd $HOME && \
    wget https://github.com/Kitware/CMake/releases/download/v3.30.7/cmake-3.30.7-linux-x86_64.sh --no-check-certificate && \
    sh cmake-3.30.7-linux-x86_64.sh --skip-license --prefix=$HOME/.local
ENV PATH="$HOME/.local/bin:$PATH"

# Install libaec from source
RUN cd $HOME && \
    git clone https://gitlab.dkrz.de/k202009/libaec.git && \
    mkdir libaec/build && \
    cmake -S $HOME/libaec -B $HOME/libaec/build -DCMAKE_INSTALL_PREFIX=$HOME/.local && \
    cd libaec/build && make -j4 && make install
ENV LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"

# Install Julia packages and extract artifact paths
RUN julia --project=@. -e 'using Pkg; Pkg.add("MPICH_jll"); Pkg.add("OpenSSL_jll")'

ENV LD_LIBRARY_PATH="$(julia --project=@. -e 'using MPICH_jll; print(MPICH_jll.artifact_dir)')/lib:$LD_LIBRARY_PATH"
ENV LD_LIBRARY_PATH="$(julia --project=@. -e 'using OpenSSL_jll; print(OpenSSL_jll.artifact_dir)')/lib:$LD_LIBRARY_PATH"

# install NCCL
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update

# /usr/lib/x86_64-linux-gnu/libnccl.so
ENV JULIA_NCCL_PATH="/usr/lib/x86_64-linux-gnu/"

# Install Legate.jl and cuNumeric.jl
RUN julia --project=. -e 'using Pkg; Pkg.add(url = "https://github.com/JuliaLegate/Legate.jl", rev = "main")'
RUN julia --project=. -e 'using Pkg; Pkg.add(url = "https://github.com/JuliaLegate/cuNumeric.jl", rev = "cuda-jl-tasking")'
RUN julia --project=. -e 'using Pkg; Pkg.build()'

RUN echo "DONE"

WORKDIR $HOME
