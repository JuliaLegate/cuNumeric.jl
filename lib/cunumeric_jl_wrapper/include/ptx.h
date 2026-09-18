#pragma once

#include "legate.h"

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <array>
#include <cstdint>
#include <string>

namespace ufi {
// ABI matches Julia CuStridedDeviceArray; strides count elements, not bytes.
template <int D>
struct CuStridedDeviceArray {
  void* ptr;
  int64_t maxsize;
  std::array<int64_t, D> dims;
  std::array<int64_t, D> strides;
  int64_t length;
};

// Shared by all PTX task launchers; modules remain in the existing
// processor-local cache populated by LoadPTXTask.
CUfunction lookup_ptx(const std::string& name, cudaStream_t stream);
}  // namespace ufi
#endif
