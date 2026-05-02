set -euo pipefail

# Default to OFF: CUDA support enabled.
# Set NO_CUDA=ON to skip CUDA.
NO_CUDA=${NO_CUDA:-OFF}

if [[ $# -ne 8 ]]; then
    echo "Usage: $0 <cunumeric-pkg> <cupynumeric-root> <legate-root> <blas-root> <cuda-include-dir> <libcuda-so> <install-dir> <nthreads>"
    exit 1
fi

CUNUMERICJL_ROOT_DIR=$1
CUPYNUMERIC_ROOT_DIR=$2
LEGATE_ROOT_DIR=$3
BLAS_LIB_DIR=$4
CUDA_DRIVER_INCLUDE_DIR=$5
CUDA_DRIVER_LIBRARY=$6
INSTALL_DIR=$7
NTHREADS=$8

if [[ ! -d "$CUNUMERICJL_ROOT_DIR" ]]; then
    echo "Error: '$CUNUMERICJL_ROOT_DIR' is not a valid directory."
    exit 1
fi

if [[ ! -d "$CUPYNUMERIC_ROOT_DIR" ]]; then
    echo "Error: '$CUPYNUMERIC_ROOT_DIR' is not a valid directory."
    exit 1
fi

if [[ ! -d "$LEGATE_ROOT_DIR" ]]; then
    echo "Error: '$LEGATE_ROOT_DIR' is not a valid directory."
    exit 1
fi

if [[ ! -d "$BLAS_LIB_DIR" ]]; then
    echo "Error: '$BLAS_LIB_DIR' is not a valid directory."
    exit 1
fi

if [[ ! -f "$BLAS_LIB_DIR/libopenblas.so" ]]; then
    echo "Error: '$BLAS_LIB_DIR/libopenblas.so' does not exist."
    exit 1
fi

if [[ "$NO_CUDA" != "ON" ]]; then
    if [[ ! -f "$CUDA_DRIVER_INCLUDE_DIR/cuda.h" ]]; then
        echo "Error: '$CUDA_DRIVER_INCLUDE_DIR/cuda.h' does not exist."
        exit 1
    fi

    if [[ ! -f "$CUDA_DRIVER_LIBRARY" ]]; then
        echo "Error: '$CUDA_DRIVER_LIBRARY' does not exist."
        exit 1
    fi
fi

CUNUMERIC_WRAPPER_SOURCE="$CUNUMERICJL_ROOT_DIR/lib/cunumeric_jl_wrapper"
BUILD_DIR="$CUNUMERIC_WRAPPER_SOURCE/build"

mkdir -p "$BUILD_DIR"
mkdir -p "$INSTALL_DIR"

echo "LEGATE_ROOT_DIR: $LEGATE_ROOT_DIR"
echo "NO_CUDA: $NO_CUDA"

if [[ "$NO_CUDA" != "ON" ]]; then
    echo "CUDA driver include dir: $CUDA_DRIVER_INCLUDE_DIR"
    echo "CUDA driver library:     $CUDA_DRIVER_LIBRARY"
fi

echo "Configuring project..."

cmake -S "$CUNUMERIC_WRAPPER_SOURCE" -B "$BUILD_DIR" \
    -D BINARYBUILDER=OFF \
    -D NOCUDA="$NO_CUDA" \
    -D CUDA_DRIVER_INCLUDE_DIR="$CUDA_DRIVER_INCLUDE_DIR" \
    -D CUDA_DRIVER_LIBRARY="$CUDA_DRIVER_LIBRARY" \
    -D CMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -D CMAKE_PREFIX_PATH="$CUPYNUMERIC_ROOT_DIR;$LEGATE_ROOT_DIR;" \
    -D CUPYNUMERIC_PATH="$CUPYNUMERIC_ROOT_DIR" \
    -D BLAS_LIBRARIES="$BLAS_LIB_DIR/libopenblas.so" \
    -D PROJECT_INSTALL_PATH="$INSTALL_DIR" \
    -D CMAKE_BUILD_TYPE=Release

cmake --build "$BUILD_DIR" --parallel "$NTHREADS" --verbose
