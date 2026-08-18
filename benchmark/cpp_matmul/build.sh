legate_root=`python -c 'import legate.install_info as i; from pathlib import Path; print(Path(i.libpath).parent.resolve())'`
echo "Using Legate at $legate_root"
cupynumeric_root=`python -c 'import cupynumeric.install_info as i; from pathlib import Path; print(Path(i.libpath).parent.resolve())'`
echo "Using cuPyNumeric at $cupynumeric_root"

cmake -S . -B build -D legate_ROOT="$legate_root" -D cupynumeric_ROOT="$cupynumeric_root" -D CMAKE_BUILD_TYPE=Debug
cmake --build build --parallel 8 --verbose
