/* Copyright 2025 Northwestern University,
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
 */

#include "cuda.h"

#include <regex>

#include "cupynumeric.h"
#include "legate.h"
#include "legate/utilities/proc_local_storage.h"
#include "legion.h"
#include "ufi.h"

#define KERNEL_NAME_BUFFER_SIZE 100

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

namespace ufi {
using namespace Legion;

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

std::string context_to_string(CUcontext ctx) {
  std::ostringstream oss;
  oss << ctx;  // prints pointer value
  return oss.str();
}

std::string key_to_string(const FunctionKey &key) {
  return "CUcontext: " + context_to_string(key.first) + ", kernel: \"" +
         key.second + "\"";
}

using FunctionMap = std::unordered_map<FunctionKey, CUfunction, FunctionKeyHash,
                                       FunctionKeyEqual>;

static legate::ProcLocalStorage<FunctionMap> cufunction_ptr{};

// static legate::ProcLocalStorage<uint64_t> cufunction_ptr{};

// https://github.com/nv-legate/legate.pandas/blob/branch-22.01/src/udf/eval_udf_gpu.cc
/*static*/ void RunPTXTask::gpu_variant(legate::TaskContext context) {
  cudaStream_t stream_ = context.get_task_stream();
  std::string kernel_name = context.scalar(0).value<std::string>();
  CUcontext ctx;
  cuStreamGetCtx(stream_, &ctx);

  FunctionKey key = {ctx, kernel_name};
  assert(cufunction_ptr.has_value() && "[RunPTXTask] hashmap has no data.");
  FunctionMap &fmap = cufunction_ptr.get();

  auto it = fmap.find(key);

  if (it == fmap.end()) {
    // for DEBUG output
    std::cerr << "[RunPTXTask] Could not find key: " << key_to_string(key)
              << std::endl;
    for (const auto &[k, v] : fmap) {
      std::cerr << "[RunPTXTask] Map key: " << key_to_string(k) << std::endl;
    }
    assert(0 && "[RunPTXTask] key is not found in hashmap");
  }
  CUfunction func = it->second;  // second arg is function ptr.

  uint32_t padded_bytes = 16;
  uint32_t N = context.scalar(1).value<uint32_t>();

  void *a = (void *)context.input(0).data().read_accessor<float, 1>().ptr(
      Realm::Point<1>(0));
  void *b = (void *)context.input(1).data().read_accessor<float, 1>().ptr(
      Realm::Point<1>(0));
  void *c = (void *)context.output(0).data().write_accessor<float, 1>().ptr(
      Realm::Point<1>(0));

  uint32_t THREADS_PER_BLOCK = 256;

  const uint32_t gridDimX = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  const uint32_t gridDimY = 1;
  const uint32_t gridDimZ = 1;

  const uint32_t blockDimX = THREADS_PER_BLOCK;
  const uint32_t blockDimY = 1;
  const uint32_t blockDimZ = 1;

  struct CuDeviceArray {
    void *ptr;         // device pointer to data
    int64_t maxsize;   // total allocated size in bytes
    int64_t length;    // length of the 1D array (number of elements)
    int64_t reserved;  // reserved or padding field, set to 0
  };

  size_t buffer_size =
      padded_bytes + (3) * sizeof(CuDeviceArray) + context.scalar(1).size();

  std::vector<char> arg_buffer(buffer_size);
  char *raw_arg_buffer = arg_buffer.data();

  auto p = raw_arg_buffer;

  p += padded_bytes;

  //   *reinterpret_cast<const void **>(p) = a;
  //   p += sizeof(void *);
  //   *reinterpret_cast<const void **>(p) = b;
  //   p += sizeof(void *);
  //   *reinterpret_cast<void **>(p) = c;
  //   p += sizeof(void *);

  //   memcpy(p, context.scalar(0).ptr(), context.scalar(0).size());

  CuDeviceArray a_desc = {a, N * sizeof(float), N, 0};
  CuDeviceArray b_desc = {b, N * sizeof(float), N, 0};
  CuDeviceArray c_desc = {c, N * sizeof(float), N, 0};

  memcpy(p, &a_desc, sizeof(CuDeviceArray));
  p += sizeof(CuDeviceArray);
  memcpy(p, &b_desc, sizeof(CuDeviceArray));
  p += sizeof(CuDeviceArray);
  memcpy(p, &c_desc, sizeof(CuDeviceArray));
  p += sizeof(CuDeviceArray);

  memcpy(p, &N, sizeof(uint32_t));

  void *config[] = {
      CU_LAUNCH_PARAM_BUFFER_POINTER,
      static_cast<void *>(raw_arg_buffer),
      CU_LAUNCH_PARAM_BUFFER_SIZE,
      &buffer_size,
      CU_LAUNCH_PARAM_END,
  };

  TEST_PRINT_DEBUG(a, N, float, "%f", stream_, "array a");
  TEST_PRINT_DEBUG(b, N, float, "%f", stream_, "array b");
  TEST_PRINT_DEBUG(c, N, float, "%f", stream_, "array c");

  fprintf(stderr, "N is  %u\n", N);
  fprintf(stderr, "running function :%p\n", func);

  CUstream custream_ = reinterpret_cast<CUstream>(stream_);
  DRIVER_ERROR_CHECK(cuLaunchKernel(func, gridDimX, gridDimY, gridDimZ,
                                    blockDimX, blockDimY, blockDimZ, 0,
                                    custream_, NULL, config));

  DRIVER_ERROR_CHECK(cuStreamSynchronize(stream_));
  TEST_PRINT_DEBUG(c, N, float, "%f", stream_, "after array c");
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

  std::cerr << ptx << std::endl;

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

  fprintf(stderr, "placed function :%p\n", hfunc);
}
}  // namespace ufi

// https://github.com/nv-legate/cupynumeric/blob/7e554b576ccc2d07a86986949cea79e56c690fe1/src/cupynumeric/ndarray.cc#L2120
// Copied method from the above link.
legate::LogicalStore broadcast(const std::vector<uint64_t> &shape,
                               legate::LogicalStore &store) {
  int32_t diff = static_cast<int32_t>(shape.size()) - store.dim();

  auto result = store;
  for (int32_t dim = 0; dim < diff; ++dim) {
    result = result.promote(dim, shape[dim]);
  }

  std::vector<uint64_t> orig_shape = result.extents().data();
  for (uint32_t dim = 0; dim < shape.size(); ++dim) {
    if (orig_shape[dim] != shape[dim]) {
      result = result.project(dim, 0).promote(dim, shape[dim]);
    }
  }

  return result;
}

legate::Library get_lib() {
  auto runtime = cupynumeric::CuPyNumericRuntime::get_runtime();
  return runtime->get_library();
}

cupynumeric::NDArray new_task(std::string kernel_name,
                              cupynumeric::NDArray rhs1,
                              cupynumeric::NDArray rhs2,
                              cupynumeric::NDArray output, uint32_t N) {
  auto runtime = legate::Runtime::get_runtime();
  auto library = get_lib();
  auto task =
      runtime->create_task(library, legate::LocalTaskID{ufi::RUN_PTX_TASK});

  auto &out_shape = output.shape();
  auto rhs1_temp = rhs1.get_store();
  auto rhs2_temp = rhs2.get_store();

  auto p_lhs = task.add_output(output.get_store());
  auto p_rhs1 = task.add_input(broadcast(out_shape, rhs1_temp));
  auto p_rhs2 = task.add_input(broadcast(out_shape, rhs2_temp));

  task.add_scalar_arg(legate::Scalar(kernel_name));
  task.add_scalar_arg(legate::Scalar(N));
  task.add_constraint(legate::align(p_lhs, p_rhs1));
  task.add_constraint(legate::align(p_rhs1, p_rhs2));

  runtime->submit(std::move(task));
  return output;
}

void ptx_task(std::string ptx, std::string kernel_name) {
  auto runtime = legate::Runtime::get_runtime();
  auto library = get_lib();
  auto task =
      runtime->create_task(library, legate::LocalTaskID{ufi::LOAD_PTX_TASK});
  task.add_scalar_arg(legate::Scalar(ptx));
  task.add_scalar_arg(legate::Scalar(kernel_name));

  runtime->submit(std::move(task));
}

void register_tasks() {
  auto library = get_lib();
  ufi::LoadPTXTask::register_variants(library);
  ufi::RunPTXTask::register_variants(library);
}

void gpu_sync() {
  cudaStream_t stream_ = nullptr;
  ERROR_CHECK(cudaDeviceSynchronize());
}

std::string extract_kernel_name(std::string ptx) {
  std::cmatch line_match;
  // there should be a built in find name of ufi function - pat
  bool match = std::regex_search(ptx.c_str(), line_match,
                                 std::regex(".visible .entry [_a-zA-Z0-9$]+"));

  const auto &matched_line = line_match.begin()->str();
  auto fun_name =
      matched_line.substr(matched_line.rfind(" ") + 1, matched_line.size());
  return fun_name;
}

void wrap_cuda_methods(jlcxx::Module &mod) {
  mod.method("register_tasks", &register_tasks);
  mod.method("get_library", &get_lib);
  mod.method("new_task", &new_task);
  mod.method("ptx_task", &ptx_task);
  mod.method("gpu_sync", &gpu_sync);
  mod.method("extract_kernel_name", &extract_kernel_name);
}