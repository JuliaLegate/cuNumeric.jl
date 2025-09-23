#!/bin/bash
set -e
export LEGATE_AUTO_CONFIG=0
julia -e 'using Pkg; Pkg.test("cuNumeric")'