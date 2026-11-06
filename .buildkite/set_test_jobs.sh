#!/usr/bin/env bash

# Source on the agent running the tests, not the pipeline-upload agent.
# All CUDA agents advertise JULIA_CPU_THREADS=2 despite different capacities.
# ParallelTestRunner reads PTR_NUM_JOBS when --jobs is not specified.
export PTR_NUM_JOBS=8
if [[ "${BUILDKITE_AGENT_NAME:-}" == "gpuci1" ]]; then
    export PTR_NUM_JOBS=2
fi

echo "Running GPU tests with $PTR_NUM_JOBS parallel workers"
