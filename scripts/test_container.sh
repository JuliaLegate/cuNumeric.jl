#!/bin/bash
set -e
export LEGATE_AUTO_CONFIG=0
export LEGATE_SHOW_CONFIG=1
export LEGATE_CONFIG="--cpus 1 --utility 1 --sysmem 4000"

julia -e 'using Pkg; Pkg.test("cuNumeric")'