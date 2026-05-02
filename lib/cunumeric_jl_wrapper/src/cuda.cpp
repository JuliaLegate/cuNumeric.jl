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

#include "cuda_macros.h"
#include "legate.h"
#include "legate/utilities/proc_local_storage.h"
#include "legion.h"
#include "types.h"
#include "ufi.h"

#define CUDA_DEBUG 0

#define BLOCK_START 1
#define THREAD_START 4
#define ARG_OFFSET 7
#define PREFIX_SCALAR_COUNT_INDEX 7
#define KERNEL_INPUT_ARG_COUNT_INDEX 8
#define KERNEL_OUTPUT_ARG_COUNT_INDEX 9
#define BC_NDARRAY_PATCH_COUNT_INDEX 10
#define BC_SCALAR_PATCH_COUNT_INDEX 11
#define BC_BLOB_SIZE_INDEX 12
#define BCAST_ARG_OFFSET 13

// global padding for CUDA.jl kernel state
std::size_t padded_bytes_kernel_state = 16;

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

struct LaunchDims {
  std::uint32_t bx, by, bz;
  std::uint32_t tx, ty, tz;
};

inline LaunchDims read_launch_dims(const legate::TaskContext &context) {
  return LaunchDims{
      context.scalar(BLOCK_START + 0).value<std::uint32_t>(),
      context.scalar(BLOCK_START + 1).value<std::uint32_t>(),
      context.scalar(BLOCK_START + 2).value<std::uint32_t>(),
      context.scalar(THREAD_START + 0).value<std::uint32_t>(),
      context.scalar(THREAD_START + 1).value<std::uint32_t>(),
      context.scalar(THREAD_START + 2).value<std::uint32_t>(),
  };
}

inline std::pair<CUcontext, CUfunction> lookup_kernel_function(
    const std::string &kernel_name, cudaStream_t stream_) {
  CUcontext ctx;
  cuStreamGetCtx(stream_, &ctx);

  FunctionKey key = {ctx, kernel_name};
  assert(cufunction_ptr.has_value());
  FunctionMap &fmap = cufunction_ptr.get();
  auto it = fmap.find(key);
#ifdef CUDA_DEBUG
  if (it == fmap.end()) {
    std::cerr << "[RunPTXTask] Could not find key: " << key_to_string(key)
              << std::endl;
    for (const auto &[k, v] : fmap) {
      std::cerr << "[RunPTXTask] Map key: " << key_to_string(k) << std::endl;
    }
    assert(0 && "[RunPTXTask] key is not found in hashmap");
  }
#endif
  assert(it != fmap.end());
  return {ctx, it->second};
}

inline void append_array_args(char *&p, const legate::TaskContext &context,
                              std::size_t count, bool is_input) {
  for (std::size_t i = 0; i < count; ++i) {
    auto ps = is_input ? context.input(i) : context.output(i);
    auto code = ps.type().code();
    auto dim = ps.dim();
    legate::double_dispatch(
        dim, code, ufiFunctor{},
        is_input ? ufi::AccessMode::READ : ufi::AccessMode::WRITE, p, ps);
  }
}

inline void write_array_descriptor(char *dst, const legate::PhysicalArray &ps,
                                   bool is_input) {
  char *p = dst;
  auto code = ps.type().code();
  auto dim = ps.dim();
  legate::double_dispatch(
      dim, code, ufiFunctor{},
      is_input ? ufi::AccessMode::READ : ufi::AccessMode::WRITE, p, ps);
}

inline void launch_with_buffer(CUfunction func, const LaunchDims &dims,
                               cudaStream_t stream_,
                               std::vector<char> &arg_buffer,
                               std::size_t used_buffer_size) {
  void *config[] = {
      CU_LAUNCH_PARAM_BUFFER_POINTER,
      static_cast<void *>(arg_buffer.data()),
      CU_LAUNCH_PARAM_BUFFER_SIZE,
      &used_buffer_size,
      CU_LAUNCH_PARAM_END,
  };

  CUstream custream_ = reinterpret_cast<CUstream>(stream_);
  DRIVER_ERROR_CHECK(cuLaunchKernel(func, dims.bx, dims.by, dims.bz, dims.tx,
                                    dims.ty, dims.tz, 0, custream_, nullptr,
                                    config));
}

// https://github.com/nv-legate/legate.pandas/blob/branch-22.01/src/udf/eval_udf_gpu.cc
/*static*/ void RunPTXTask::gpu_variant(legate::TaskContext context) {
  cudaStream_t stream_ = context.get_task_stream();
  std::string kernel_name = context.scalar(0).value<std::string>();  // 0
  const LaunchDims dims = read_launch_dims(context);
  auto kernel = lookup_kernel_function(kernel_name, stream_);
  CUfunction func = kernel.second;

  const std::size_t num_inputs = context.num_inputs();
  const std::size_t num_outputs = context.num_outputs();
  const std::size_t num_scalars = context.num_scalars();
  // Layout: [kernel-state][inputs][outputs][user-scalars]
  std::size_t max_buffer_size =
      padded_bytes_kernel_state +
      (num_inputs + num_outputs) * sizeof(CuDeviceArray<REALM_MAX_DIM>);
  for (std::size_t i = ARG_OFFSET; i < num_scalars; ++i)
    max_buffer_size += context.scalar(i).size();

  std::vector<char> arg_buffer(max_buffer_size);
  char *p = arg_buffer.data() + padded_bytes_kernel_state;

  append_array_args(p, context, num_inputs, true);
  append_array_args(p, context, num_outputs, false);
  for (std::size_t i = ARG_OFFSET; i < num_scalars; ++i) {
    const auto &scalar = context.scalar(i);
    memcpy(p, scalar.ptr(), scalar.size());
    p += scalar.size();
  }

  std::size_t buffer_size = p - arg_buffer.data();  // calc used buffer

  launch_with_buffer(func, dims, stream_, arg_buffer, buffer_size);

  // DRIVER_ERROR_CHECK(cuStreamSynchronize(stream_));
}

/*
This code is specifically for the fusion of broadcast operations.
This code uses the passed offsets to patch together the Broadcasted type
that the CUDA kernel expects. The Broadcasted type has all inputs in order,
but mixed scalars and arrays. e.g., [Array, Scalar, Array, Array].
We must use the offsets to figure out which arguments should be marked as
scalars and which should be marked as inputs.
*/

/*static*/ void RunPTXBroadcastTask::gpu_variant(legate::TaskContext context) {
  cudaStream_t stream_ = context.get_task_stream();
  std::string kernel_name = context.scalar(0).value<std::string>();  // 0
  const LaunchDims dims = read_launch_dims(context);

  std::uint32_t prefix_scalar_count =
      context.scalar(PREFIX_SCALAR_COUNT_INDEX).value<std::uint32_t>();  // 7
  std::uint32_t kernel_input_args_count =
      context.scalar(KERNEL_INPUT_ARG_COUNT_INDEX).value<std::uint32_t>();  // 8
  std::uint32_t kernel_output_args_count =
      context.scalar(KERNEL_OUTPUT_ARG_COUNT_INDEX)
          .value<std::uint32_t>();  // 9
  std::uint32_t bc_ndarray_patch_count =
      context.scalar(BC_NDARRAY_PATCH_COUNT_INDEX)
          .value<std::uint32_t>();  // 10
  std::uint32_t bc_scalar_patch_count =
      context.scalar(BC_SCALAR_PATCH_COUNT_INDEX).value<std::uint32_t>();  // 11
  std::uint32_t bc_blob_size =
      context.scalar(BC_BLOB_SIZE_INDEX).value<std::uint32_t>();  // 12

  const std::size_t num_inputs = context.num_inputs();
  const std::size_t num_outputs = context.num_outputs();
  const std::size_t num_scalars = context.num_scalars();

  assert(kernel_input_args_count <= num_inputs);
  assert(kernel_output_args_count <= num_outputs);
  const std::size_t array_patch_start = BCAST_ARG_OFFSET;
  const std::size_t scalar_patch_start =
      array_patch_start + 2 * bc_ndarray_patch_count;
  const std::size_t prefix_start = scalar_patch_start + bc_scalar_patch_count;
  const std::size_t prefix_end = prefix_start + prefix_scalar_count;
  const std::size_t scalar_values_start = prefix_end;

  assert(scalar_values_start + bc_scalar_patch_count <= num_scalars);

  auto kernel = lookup_kernel_function(kernel_name, stream_);
  CUfunction func = kernel.second;

  std::size_t max_buffer_size =
      padded_bytes_kernel_state +
      (kernel_input_args_count + kernel_output_args_count) *
          sizeof(CuDeviceArray<REALM_MAX_DIM>) +
      bc_blob_size;
  for (std::size_t i = prefix_start; i < prefix_end; ++i) {
    max_buffer_size += context.scalar(i).size();
  }

  std::vector<char> arg_buffer(max_buffer_size);
  char *p = arg_buffer.data() + padded_bytes_kernel_state;

  for (std::size_t i = prefix_start; i < prefix_end; ++i) {
    const auto &scalar = context.scalar(i);
    memcpy(p, scalar.ptr(), scalar.size());
    p += scalar.size();
  }

  append_array_args(p, context, kernel_input_args_count, true);
  append_array_args(p, context, kernel_output_args_count, false);

  std::vector<char> patched_bc(bc_blob_size, 0);

  for (std::size_t j = 0; j < bc_ndarray_patch_count; ++j) {
    std::uint32_t offset =
        context.scalar(array_patch_start + 2 * j).value<std::uint32_t>();
    std::uint32_t input_index =
        context.scalar(array_patch_start + 2 * j + 1).value<std::uint32_t>();
    assert(input_index < num_inputs);
    assert(offset <= patched_bc.size());
    char *begin = patched_bc.data() + offset;
    char *end = begin;
    write_array_descriptor(end, context.input(input_index), true);
    assert(static_cast<std::size_t>(end - patched_bc.data()) <=
           patched_bc.size());
  }

  for (std::size_t j = 0; j < bc_scalar_patch_count; ++j) {
    std::uint32_t offset =
        context.scalar(scalar_patch_start + j).value<std::uint32_t>();
    const auto &scalar_value = context.scalar(scalar_values_start + j);
    assert(offset + scalar_value.size() <= patched_bc.size());
    memcpy(patched_bc.data() + offset, scalar_value.ptr(), scalar_value.size());
  }

  memcpy(p, patched_bc.data(), patched_bc.size());
  p += patched_bc.size();

  std::size_t buffer_size = p - arg_buffer.data();

  launch_with_buffer(func, dims, stream_, arg_buffer, buffer_size);
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
  mod.set_const("RUN_PTX_BROADCAST",
                legate::LocalTaskID{ufi::TaskIDs::RUN_PTX_BROADCAST_TASK});
}
