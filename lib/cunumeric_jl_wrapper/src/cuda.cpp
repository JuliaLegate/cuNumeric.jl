/* Copyright 2026 Northwestern University,
 *                   Carnegie Mellon University University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author(s): David Krasowska <krasow@u.northwestern.edu>
 *            Ethan Meitz <emeitz@andrew.cmu.edu>
 *            Nader Rahhal <naderrahhal2026@u.northwestern.edu>
 */

#include "cuda.h"

#include <cstdint>
#include <regex>

#include "legate.h"
#include "legate/utilities/proc_local_storage.h"
#include "legion.h"
#include "types.h"
#include "ufi.h"

#define CUDA_DEBUG 1

#define BLOCK_START 1
#define THREAD_START 4
#define ARG_OFFSET 7

// global padding for CUDA.jl kernel state
std::size_t padded_bytes_kernel_state = 16;

#define ERROR_CHECK(x)                                                 \
  {                                                                    \
    cudaError_t status = x;                                            \
    if (status != cudaSuccess) {                                       \
      fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status));                             \
      if (stream_) cudaStreamDestroy(stream_);                         \
      exit(-1);                                                        \
    }                                                                  \
  }

#define DRIVER_ERROR_CHECK(x)                                                 \
  {                                                                           \
    CUresult status = x;                                                      \
    if (status != CUDA_SUCCESS) {                                             \
      const char *err_str = nullptr;                                          \
      cuGetErrorString(status, &err_str);                                     \
      fprintf(stderr, "CUDA Driver Error at %s:%d: %s\n", __FILE__, __LINE__, \
              err_str);                                                       \
      if (stream_) cudaStreamDestroy(stream_);                                \
      exit(-1);                                                               \
    }                                                                         \
  }

#define TEST_PRINT_DEBUG(dev_ptr, N, T, format, stream, message)            \
  {                                                                         \
    std::vector<T> host_arr(N);                                             \
    ERROR_CHECK(cudaMemcpy(host_arr.data(),                                 \
                           reinterpret_cast<const T *>(dev_ptr),            \
                           sizeof(T) * N, cudaMemcpyDeviceToHost));         \
    ERROR_CHECK(cudaStreamSynchronize(stream));                             \
    fprintf(stderr, "[TEST_PRINT] %s: " format "\n", message, host_arr[0]); \
  }

#ifdef CUDA_DEBUG
#define CUDA_DEBUG_PRINT(x) \
  do {                      \
    x;                      \
  } while (0)
#else
#define CUDA_DEBUG_PRINT(x) \
  do {                      \
  } while (0)
#endif

namespace ufi {
using namespace Legion;
// TODO CUcontext key hashing is redundant. ProcLocalStorage is local to the
// cuContext.
using FunctionKey = std::pair<CUcontext, std::string>;

struct FunctionKeyHash {
  std::size_t operator()(const FunctionKey &k) const {
    return std::hash<CUcontext>()(k.first) ^
           (std::hash<std::string>()(k.second) << 1);
  }
};

struct FunctionKeyEqual {
  bool operator()(const FunctionKey &lhs, const FunctionKey &rhs) const {
    return lhs.first == rhs.first && lhs.second == rhs.second;
  }
};

using FunctionMap = std::unordered_map<FunctionKey, CUfunction, FunctionKeyHash,
                                       FunctionKeyEqual>;

static legate::ProcLocalStorage<FunctionMap> cufunction_ptr{};

#ifdef CUDA_DEBUG
std::string context_to_string(CUcontext ctx) {
  std::ostringstream oss;
  oss << ctx;  // prints pointer value
  return oss.str();
}

std::string key_to_string(const FunctionKey &key) {
  return "CUcontext: " + context_to_string(key.first) + ", kernel: \"" +
         key.second + "\"";
}
#endif

enum class AccessMode {
  READ,
  WRITE,
};

template <size_t D>
struct CuDeviceArray {
  void *ptr;                     // Pointer to device memory
  uint64_t maxsize;              // Total allocated size in bytes
  std::array<uint64_t, D> dims;  // Fixed-size array of dimension sizes
  uint64_t length;               // Number of elements (at the end)
};

#define CUDA_DEVICE_ARRAY_ARG(MODE, ACCESSOR_CALL)                             \
  template <                                                                   \
      typename T, int D,                                                       \
      typename std::enable_if<(D >= 1 && D <= REALM_MAX_DIM), int>::type = 0>  \
  void cuda_device_array_arg_##MODE(char *&p,                                  \
                                    const legate::PhysicalArray &rf) {         \
    auto shp = rf.shape<D>();                                                  \
    auto acc = rf.data().ACCESSOR_CALL<T, D>();                                \
    CUDA_DEBUG_PRINT(std::cerr << "[RunPTXTask] " #MODE " accessor shape: "    \
                               << shp.lo << " - " << shp.hi << ", dim: " << D  \
                               << std::endl;                                   \
                     std::cerr << "[RunPTXTask] " #MODE " accessor strides: "  \
                               << acc.accessor.strides << std::endl;);         \
    void *dev_ptr = const_cast<void *>(/*.lo to ensure multiple GPU support*/  \
                                       static_cast<const void *>(              \
                                           acc.ptr(Realm::Point<D>(shp.lo)))); \
    auto extents = shp.hi - shp.lo + legate::Point<D>::ONES();                 \
    CuDeviceArray<D> desc;                                                     \
    desc.ptr = dev_ptr;                                                        \
    desc.maxsize = shp.volume() * sizeof(T);                                   \
    for (size_t i = 0; i < D; ++i) {                                           \
      desc.dims[i] = extents[i];                                               \
    }                                                                          \
    desc.length = shp.volume();                                                \
    memcpy(p, &desc, sizeof(CuDeviceArray<D>));                                \
    p += sizeof(CuDeviceArray<D>);                                             \
  }

CUDA_DEVICE_ARRAY_ARG(read, read_accessor);    // cuda_device_array_arg_read
CUDA_DEVICE_ARRAY_ARG(write, write_accessor);  // cuda_device_array_arg_write

struct ufiFunctor {
  template <legate::Type::Code CODE, int DIM>
  void operator()(AccessMode mode, char *&p, const legate::PhysicalArray &arr) {
    using CppT = typename legate_util::code_to_cxx<CODE>::type;
    if (mode == AccessMode::READ)
      cuda_device_array_arg_read<CppT, DIM>(p, arr);
    else
      cuda_device_array_arg_write<CppT, DIM>(p, arr);
  }
};

// https://github.com/nv-legate/legate.pandas/blob/branch-22.01/src/udf/eval_udf_gpu.cc
/*static*/ void RunPTXTask::gpu_variant(legate::TaskContext context) {
  cudaStream_t stream_ = context.get_task_stream();
  std::string kernel_name = context.scalar(0).value<std::string>();  // 0

  std::uint32_t bx =
      context.scalar(BLOCK_START + 0).value<std::uint32_t>();  // 1
  std::uint32_t by =
      context.scalar(BLOCK_START + 1).value<std::uint32_t>();  // 2
  std::uint32_t bz =
      context.scalar(BLOCK_START + 2).value<std::uint32_t>();  // 3

  std::uint32_t tx =
      context.scalar(THREAD_START + 0).value<std::uint32_t>();  // 4
  std::uint32_t ty =
      context.scalar(THREAD_START + 1).value<std::uint32_t>();  // 5
  std::uint32_t tz =
      context.scalar(THREAD_START + 2).value<std::uint32_t>();  // 6

  CUcontext ctx;
  cuStreamGetCtx(stream_, &ctx);

  FunctionKey key = {ctx, kernel_name};
  assert(cufunction_ptr.has_value());
  FunctionMap &fmap = cufunction_ptr.get();

  auto it = fmap.find(key);

#ifdef CUDA_DEBUG
  if (it == fmap.end()) {
    // for DEBUG output
    std::cerr << "[RunPTXTask] Could not find key: " << key_to_string(key)
              << std::endl;
    for (const auto &[k, v] : fmap) {
      std::cerr << "[RunPTXTask] Map key: " << key_to_string(k) << std::endl;
    }
    assert(0 && "[RunPTXTask] key is not found in hashmap");
  }
#endif

  assert(it != fmap.end());
  CUfunction func = it->second;

  const std::size_t num_inputs = context.num_inputs();
  const std::size_t num_outputs = context.num_outputs();
  const std::size_t num_scalars = context.num_scalars();
  const std::size_t num_reductions =
      context.num_reductions();  // unused for now

  // compute total size: all device arrays + all scalars
  // skip scalar 0-2 (kernel_name, threads, blocks)
  // we allocate extra to decrease looping and dynamic dispatching on dim
  std::size_t max_buffer_size =
      padded_bytes_kernel_state +
      (num_inputs + num_outputs) * sizeof(CuDeviceArray<REALM_MAX_DIM>);
  for (std::size_t i = ARG_OFFSET; i < num_scalars; ++i)
    max_buffer_size += context.scalar(i).size();

  std::vector<char> arg_buffer(max_buffer_size);
  char *p = arg_buffer.data() + padded_bytes_kernel_state;

  for (std::size_t i = 0; i < num_inputs; ++i) {
    auto ps = context.input(i);
    auto code = ps.type().code();
    auto dim = ps.dim();
#ifdef CUDA_DEBUG
    std::cerr << "[RunPTXTask] Input " << i << " type: " << code
              << ", dim: " << dim << std::endl;
#endif
    // dispatch on dim and code with ufiFunctor operator()
    legate::double_dispatch(dim, code, ufiFunctor{}, ufi::AccessMode::READ, p,
                            ps);
  }
  for (std::size_t i = 0; i < num_outputs; ++i) {
    auto ps = context.output(i);
    auto code = ps.type().code();
    auto dim = ps.dim();
#ifdef CUDA_DEBUG
    std::cerr << "[RunPTXTask] Output " << i << " type: " << code
              << ", dim: " << dim << std::endl;
#endif
    legate::double_dispatch(dim, code, ufiFunctor{}, ufi::AccessMode::WRITE, p,
                            ps);
  }
  for (std::size_t i = ARG_OFFSET; i < num_scalars; ++i) {
    const auto &scalar = context.scalar(i);
    memcpy(p, scalar.ptr(), scalar.size());
    p += scalar.size();
  }

  std::size_t buffer_size = p - arg_buffer.data();  // calc used buffer

  void *config[] = {
      CU_LAUNCH_PARAM_BUFFER_POINTER,
      static_cast<void *>(arg_buffer.data()),
      CU_LAUNCH_PARAM_BUFFER_SIZE,
      &buffer_size,
      CU_LAUNCH_PARAM_END,
  };

  CUstream custream_ = reinterpret_cast<CUstream>(stream_);

#ifdef CUDA_DEBUG
  std::cerr << "[RunPTXTask] Launching kernel " << kernel_name
            << " with blocks (" << bx << "," << by << "," << bz
            << ") and threads (" << tx << "," << ty << "," << tz << ")"
            << " on CUcontext " << context_to_string(ctx) << std::endl;
#endif
  // Launch the kernel
  DRIVER_ERROR_CHECK(cuLaunchKernel(func, bx, by, bz, tx, ty, tz, 0, custream_,
                                    nullptr, config));

  // DRIVER_ERROR_CHECK(cuStreamSynchronize(stream_));
}

// https://github.com/nv-legate/legate.pandas/blob/branch-22.01/src/udf/load_ptx.cc
/*static*/ void LoadPTXTask::gpu_variant(legate::TaskContext context) {
  std::string ptx = context.scalar(0).value<std::string>();
  std::string kernel_name = context.scalar(1).value<std::string>();

  cudaStream_t stream_ = context.get_task_stream();
  CUcontext ctx;
  cuStreamGetCtx(stream_, &ctx);

  FunctionKey key = std::make_pair(ctx, kernel_name);

  FunctionMap &fmap = [&]() -> FunctionMap & {
    if (cufunction_ptr.has_value()) {
      return cufunction_ptr.get();
    } else {
      cufunction_ptr.emplace(FunctionMap{});
      return cufunction_ptr.get();
    }
  }();

  auto it = fmap.find(key);
  if (!(it == fmap.end())) {
    return;
  }  // we have this exact kernel already compiled.
#ifdef CUDA_DEBUG
  std::cerr << ptx << std::endl;
#endif

  const unsigned num_options = 4;
  const size_t buffer_size = 16384;
  std::vector<char> log_info_buffer(buffer_size);
  std::vector<char> log_error_buffer(buffer_size);
  CUjit_option jit_options[] = {
      CU_JIT_INFO_LOG_BUFFER,
      CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
      CU_JIT_ERROR_LOG_BUFFER,
      CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
  };
  void *option_vals[] = {
      static_cast<void *>(log_info_buffer.data()),
      reinterpret_cast<void *>(buffer_size),
      static_cast<void *>(log_error_buffer.data()),
      reinterpret_cast<void *>(buffer_size),
  };

  CUmodule module;
  CUresult result =
      cuModuleLoadDataEx(&module, static_cast<const void *>(ptx.c_str()),
                         num_options, jit_options, option_vals);
  if (result != CUDA_SUCCESS) {
    if (result == CUDA_ERROR_OPERATING_SYSTEM) {
      fprintf(stderr,
              "ERROR: Device side asserts are not supported by the "
              "CUDA driver for MAC OSX, see NVBugs 1628896.\n");
      exit(-1);
    } else if (result == CUDA_ERROR_NO_BINARY_FOR_GPU) {
      fprintf(
          stderr,
          "ERROR: The binary was compiled for the wrong GPU architecture.\n");
      exit(-1);
    } else {
      fprintf(stderr, "Failed to load CUDA module! Error log: %s\n",
              log_error_buffer.data());
#if CUDA_VERSION >= 6050
      const char *name, *str;
      assert(cuGetErrorName(result, &name) == CUDA_SUCCESS);
      assert(cuGetErrorString(result, &str) == CUDA_SUCCESS);
      fprintf(stderr, "CU: cuModuleLoadDataEx = %d (%s): %s\n", result, name,
              str);
#else
      fprintf(stderr, "CU: cuModuleLoadDataEx = %d\n", result);
#endif
      exit(-1);
    }
  }

  CUfunction hfunc;
  result = cuModuleGetFunction(&hfunc, module, kernel_name.c_str());
  assert(result == CUDA_SUCCESS);

  fmap[key] = hfunc;

#ifdef CUDA_DEBUG
  fprintf(stderr, "placed function :%p\n", hfunc);
#endif
}
}  // namespace ufi

inline void add_xyz_scalars(legate::AutoTask &task,
                            const std::vector<uint32_t> &v) {
  uint32_t xyz[3] = {1, 1, 1};
  const size_t n = std::min<size_t>(3, v.size());
  for (size_t i = 0; i < n; ++i) xyz[i] = v[i];

  task.add_scalar_arg(legate::Scalar(xyz[0]));
  task.add_scalar_arg(legate::Scalar(xyz[1]));
  task.add_scalar_arg(legate::Scalar(xyz[2]));
}

void gpu_sync() {
  cudaStream_t stream_ = nullptr;
  ERROR_CHECK(cudaDeviceSynchronize());
}

std::string extract_kernel_name(std::string ptx) {
  std::cmatch line_match;
  bool match = std::regex_search(ptx.c_str(), line_match,
                                 std::regex(".visible .entry [_a-zA-Z0-9$]+"));

  const auto &matched_line = line_match.begin()->str();
  auto fun_name =
      matched_line.substr(matched_line.rfind(" ") + 1, matched_line.size());
  return fun_name;
}

void register_kernel_state_size(uint64_t st_size) {
  // update global
  padded_bytes_kernel_state = st_size;
}

void wrap_cuda_methods(jlcxx::Module &mod) {
  mod.method("add_xyz_scalars", &add_xyz_scalars);
  mod.method("register_kernel_state_size", &register_kernel_state_size);
  mod.method("gpu_sync", &gpu_sync);
  mod.method("extract_kernel_name", &extract_kernel_name);
  mod.set_const("LOAD_PTX", legate::LocalTaskID{ufi::TaskIDs::LOAD_PTX_TASK});
  mod.set_const("RUN_PTX", legate::LocalTaskID{ufi::TaskIDs::RUN_PTX_TASK});
}
