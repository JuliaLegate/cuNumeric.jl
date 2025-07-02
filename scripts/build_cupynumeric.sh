set -e

# Check if exactly one argument is provided
if [[ $# -ne 7 ]]; then
    echo "Usage: $0  <root-dir> <legate-dir> <nccl-dir> <cutensor-dir> <install-dir> <version> <nthreads>"
    exit 1
fi
CUNUMERIC_ROOT_DIR=$1
LEGATE_ROOT_DIR=$2
NCCL_ROOT_DIR=$3
CUTENSOR_ROOT_DIR=$4
INSTALL_DIR=$5
VERSION=$6
NTHREADS=$7

# Check if the provided argument is a valid directory
if [[ ! -d "$CUNUMERIC_ROOT_DIR" ]]; then
    echo "Error: '$CUNUMERIC_ROOT_DIR' is not a valid directory."
    exit 1
fi

if [[ ! -d "$INSTALL_DIR" ]]; then
    echo "Error: '$INSTALL_DIR' is not a valid directory."
    exit 1
fi

LAPACKE_INSTALL_DIR=$CUNUMERIC_ROOT_DIR/lapacke
if [ -d "$LAPACKE_INSTALL_DIR" ]; then
    echo "Directory '$LAPACKE_INSTALL_DIR' already exists. Skipping LAPACKE build."
else
    LAPACKE_VERSION=3.12.1
    LAPACKE_BUILD_DIR=$CUNUMERIC_ROOT_DIR/deps/lapacke_build

    mkdir -p $LAPACKE_INSTALL_DIR
    mkdir -p $LAPACKE_BUILD_DIR
    cd $LAPACKE_BUILD_DIR

    # Download LAPACK source (LAPACKe is part of it)
    curl -L -O https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v${LAPACKE_VERSION}.tar.gz
    tar -xzf v${LAPACKE_VERSION}.tar.gz
    cd lapack-${LAPACKE_VERSION}

    # Build only LAPACKe
    mkdir build && cd build
    cmake .. \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_INSTALL_PREFIX=$LAPACKE_INSTALL_DIR \
    -DBUILD_SHARED_LIBS=ON \
    -DLAPACKE=ON \
    -DBUILD_TESTING=OFF

    make -j $NTHREADS
    make install
    cd $CUNUMERIC_ROOT_DIR
fi

TAG="v$VERSION"
REPO_URL="https://github.com/nv-legate/cupynumeric"
TAG_URL="$REPO_URL/releases/tag/$TAG"
CLONE_DIR="$CUNUMERIC_ROOT_DIR/deps/cupynumeric-$VERSION"

# echo "Checking if tag $TAG exists on GitHub..."

# if curl --silent --head --fail "$TAG_URL" > /dev/null; then
#     echo "Tag $TAG exists. Cloning..."
# else
#     echo "Error: Tag $TAG does not exist at $TAG_URL"
#     exit 1
# fi

if [ -d "$CLONE_DIR" ]; then
    echo "Directory '$CLONE_DIR' already exists. Skipping clone."
else
    # git clone --branch "$TAG" --depth 1 "$REPO_URL.git" "$CLONE_DIR"
    git clone --branch "branch-25.05" --depth 1 "$REPO_URL.git" "$CLONE_DIR"
    echo "Cloned cuNumeric $VERSION into $CLONE_DIR"
fi

echo $LEGATE_ROOT_DIR

export CXXFLAGS="-I$LAPACKE_INSTALL_DIR/include" 
export LDFLAGS="-L$LAPACKE_INSTALL_DIR/lib"

BUILD_DIR=$CUNUMERIC_ROOT_DIR/deps/cupynumeric-build
cmake -S $CLONE_DIR -B $BUILD_DIR \
    -D legate_ROOT=$LEGATE_ROOT_DIR \
    -D NCCL_ROOT=$NCCL_ROOT_DIR \
    -D cutensor_ROOT=$CUTENSOR_ROOT_DIR \
 
cmake --build $BUILD_DIR  --parallel $NTHREADS --verbose
cmake --install $BUILD_DIR --prefix $INSTALL_DIR

cp $BUILD_DIR/cupynumeric-config*.cmake $INSTALL_DIR/lib/cmake/cupynumeric/
cp $BUILD_DIR/cupynumeric-targets.cmake $INSTALL_DIR/lib/cmake/cupynumeric/
cp $BUILD_DIR/cupynumeric-dependencies.cmake $INSTALL_DIR/lib/cmake/cupynumeric/
cp $BUILD_DIR/Findtblis.cmake $INSTALL_DIR/lib/cmake/cupynumeric/
