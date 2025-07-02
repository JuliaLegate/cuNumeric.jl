#!/usr/bin/env bash

if [ -z "$CUNUMERIC_JL_HOME" ]; then
  echo "Error: CUNUMERIC_JL_HOME is not set."
  exit 1
fi


CPP_TEST=$CUNUMERIC_JL_HOME/build/matmulfp32

if [ ! -f $CPP_TEST ]; then
    echo "Error: File '$CPP_TEST' does not exist. You must use build.sh OR julia build system" >&2
    exit 1
fi

# run julia
nsys profile --output=./julia-profile julia profile_matmul.jl

# run cpp 
CPP_TEST=$CUNUMERIC_JL_HOME/build/matmulfp32
echo $CPP_TEST
nsys profile --output=./cpp-profile $CPP_TEST

# python
nsys profile --output=./python-profile python profile_matmul.py
