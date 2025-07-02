#!/bin/bash

rm -rf build && mkdir -p build && cd build
cmake ".."
cmake --build . --config Release