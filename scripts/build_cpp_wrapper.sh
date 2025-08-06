set -e

# Check if exactly one argument is provided
if [[ $# -ne 8 ]]; then
    echo "Usage: $0 <cunumeric-pkg> <cupynumeric-root> <legate-root> <hdf5-root> <blas-lib> <install-dir> <branch> <nthreads>"
    exit 1
fi
CUNUMERICJL_ROOT_DIR=$1 # this is the repo root of cunumeric.jl
CUPYNUMERIC_ROOT_DIR=$2
LEGATE_ROOT_DIR=$3
HDF5_ROOT_DIR=$4
BLAS_LIB_DIR=$5
INSTALL_DIR=$6
WRAPPER_BRANCH=$7
NTHREADS=$8

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

if [[ ! -d "$HDF5_ROOT_DIR" ]]; then
    echo "Error: '$HDF5_ROOT_DIR' is not a valid directory."
    exit 1
fi

echo "Checking out wrapper branch: $WRAPPER_BRANCH"
GIT_REPO="https://github.com/JuliaLegate/cunumeric_jl_wrapper"
CUNUMERIC_WRAPPER_SOURCE=$CUNUMERICJL_ROOT_DIR/deps/cunumeric_jl_wrapper_src

if [ ! -d "$CUNUMERIC_WRAPPER_SOURCE" ]; then
    git clone $GIT_REPO $CUNUMERIC_WRAPPER_SOURCE
fi

cd "$CUNUMERIC_WRAPPER_SOURCE" || exit 1
git fetch --tags

cd "$LEGATE_WRAPPER_SOURCE" || exit 1
echo "Current repo: $(basename $(pwd))"
git remote -v

git fetch origin "$WRAPPER_BRANCH"
git checkout "$WRAPPER_BRANCH"

BUILD_DIR=$CUNUMERIC_WRAPPER_SOURCE/build

if [[ ! -d "$BUILD_DIR" ]]; then
    mkdir $BUILD_DIR 
fi

if [[ ! -d "$INSTALL_DIR" ]]; then
    mkdir $INSTALL_DIR 
fi
# patch the cmake for our custom install
diff -u $CUNUMERIC_WRAPPER_SOURCE/CMakeLists.txt $CUNUMERICJL_ROOT_DIR/deps/CMakeLists.txt > deps_install.patch  || true
cd $CUNUMERIC_WRAPPER_SOURCE
patch -i $CUNUMERIC_WRAPPER_SOURCE/deps_install.patch

echo $LEGATE_ROOT_DIR
LEGION_CMAKE_DIR=$LEGATE_ROOT_DIR/share/Legion/cmake
cmake -S $CUNUMERIC_WRAPPER_SOURCE -B $BUILD_DIR \
    -D CMAKE_PREFIX_PATH="$CUPYNUMERIC_ROOT_DIR;$LEGION_CMAKE_DIR;$LEGATE_ROOT_DIR;" \
    -D CUPYNUMERIC_PATH="$CUPYNUMERIC_ROOT_DIR" \
    -D LEGATE_PATH=$LEGATE_ROOT_DIR \
    -D HDF5_PATH=$HDF5_ROOT_DIR \
    -D BLAS_LIBRARIES=$BLAS_LIB_DIR/libopenblas.so \
    -D PROJECT_INSTALL_PATH=$INSTALL_DIR \
    -D CMAKE_BUILD_TYPE=Release
cmake --build $BUILD_DIR  --parallel $NTHREADS --verbose
cmake --install $BUILD_DIR
