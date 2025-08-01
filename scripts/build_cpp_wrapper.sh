set -e

# Check if exactly one argument is provided
if [[ $# -ne 6 ]]; then
    echo "Usage: $0 <cunumeric-pkg> <cupynumeric-dir> <legate-root> <hdf5-root> <install-dir> <nthreads>"
    exit 1
fi
CUNUMERICJL_ROOT_DIR=$1 # this is the repo root of cunumeric.jl
CUPYNUMERIC_ROOT_DIR=$2
LEGATE_ROOT_DIR=$3
HDF5_ROOT_DIR=$4
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

if [[ ! -d "$HDF5_ROOT_DIR" ]]; then
    echo "Error: '$HDF5_ROOT_DIR' is not a valid directory."
    exit 1
fi

GIT_REPO="https://github.com/JuliaLegate/cunumeric_jl_wrapper"
# COMMIT_HASH="f00bd063be66b735fc6040b40027669337399a06"
CUNUMERIC_WRAPPER_SOURCE=$LEGATEJL_PKG_ROOT_DIR/deps/cunumeric_jl_wrapper
BUILD_DIR=$CUNUMERIC_WRAPPER_SOURCE/build

if [ ! -d "$CUNUMERIC_WRAPPER_SOURCE" ]; then
    cd $CUNUMERICJL_ROOT_DIR/deps
    git clone $GIT_REPO
fi

cd $CUNUMERIC_WRAPPER_SOURCE
git fetch --tags
# git checkout $COMMIT_HASH

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
    -D PROJECT_INSTALL_PATH=$INSTALL_DIR
    -D CMAKE_BUILD_TYPE=Release
cmake --build $BUILD_DIR  --parallel $NTHREADS --verbose
cmake --install $BUILD_DIR
