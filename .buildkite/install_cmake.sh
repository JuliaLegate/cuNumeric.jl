# shellcheck shell=bash
# Source to put a pinned CMake first on PATH (developer wrapper builds need it).
CMAKE_VERSION="3.30.7"
CMAKE_ROOT="$(mktemp -d)"
CMAKE_INSTALLER="$CMAKE_ROOT/cmake-installer.sh"
curl --fail --silent --show-error --location \
    --output "$CMAKE_INSTALLER" \
    "https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION-linux-x86_64.sh"
sh "$CMAKE_INSTALLER" --skip-license --prefix="$CMAKE_ROOT"
export PATH="$CMAKE_ROOT/bin:$PATH"
cmake --version
