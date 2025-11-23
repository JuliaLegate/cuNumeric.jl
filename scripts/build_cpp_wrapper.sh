set -e

# Check if exactly one argument is provided
if [[ $# -ne 6 ]]; then
    echo "Usage: $0 <cunumeric-pkg> <cupynumeric-root> <legate-root> <blas-lib> <install-dir> <nthreads>"
    exit 1
fi
CUNUMERICJL_ROOT_DIR=$1 # this is the repo root of cunumeric.jl
CUPYNUMERIC_ROOT_DIR=$2
LEGATE_ROOT_DIR=$3
BLAS_LIB_DIR=$4
INSTALL_DIR=$5
NTHREADS=$6

# Check if the provided argument is a valid directory

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

CUNUMERIC_WRAPPER_SOURCE=$CUNUMERICJL_ROOT_DIR/lib/cunumeric_jl_wrapper
BUILD_DIR=$CUNUMERIC_WRAPPER_SOURCE/build

if [[ ! -d "$BUILD_DIR" ]]; then
    mkdir -p $BUILD_DIR 
fi

if [[ ! -d "$INSTALL_DIR" ]]; then
    mkdir -p $INSTALL_DIR 
fi

echo $LEGATE_ROOT_DIR

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "Configuring project..."
    cmake -S "$CUNUMERIC_WRAPPER_SOURCE" -B "$BUILD_DIR" \
        -D BINARYBUILDER=OFF \
        -D CMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -D CMAKE_PREFIX_PATH="$CUPYNUMERIC_ROOT_DIR;$LEGATE_ROOT_DIR;" \
        -D CUPYNUMERIC_PATH="$CUPYNUMERIC_ROOT_DIR" \
        -D BLAS_LIBRARIES="$BLAS_LIB_DIR/libopenblas.so" \
        -D PROJECT_INSTALL_PATH="$INSTALL_DIR" \
        -D CMAKE_BUILD_TYPE=Releases
else
    echo "Skipping configure (already done in $BUILD_DIR)"
fi

cmake --build "$BUILD_DIR" --parallel "$NTHREADS" --verbose
