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

struct CN_NDArray {
  cupynumeric::NDArray obj;
};

// #define CUDA_DEBUG 1

#define BLOCK_START 1
#define THREAD_START 4
#define ARG_OFFSET 7

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
// TODO CUcontext key hashing is redundant. ProcLocalStorage is local to the
// cuContext. I didn't know that when designing this hashing method.
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

struct CuDeviceArray {
  void *ptr;
  int64_t maxsize;
  int64_t length;
  int64_t reserved;
};

enum class AccessMode {
  READ,
  WRITE,
};

/* TODO::  check if std::enable_if is reducing the template expansion */
#define CUDA_DEVICE_ARRAY_ARG(MODE, ACCESSOR_CALL)                             \
  template <                                                                   \
      typename T, int D,                                                       \
      typename std::enable_if<(D >= 1 && D <= REALM_MAX_DIM), int>::type = 0>  \
  void cuda_device_array_arg_##MODE(char *&p,                                  \
                                    const legate::PhysicalArray &rf) {         \
    auto shp = rf.shape<D>();                                                  \
    auto acc = rf.data().ACCESSOR_CALL<T, D>();                                \
    void *dev_ptr = const_cast<void *>(/*.lo to ensure multiple GPU support*/  \
                                       static_cast<const void *>(              \
                                           acc.ptr(Realm::Point<D>(shp.lo)))); \
                                                                               \
    CuDeviceArray desc = {                                                     \
        dev_ptr, static_cast<int64_t>(shp.volume()) * (int64_t)sizeof(T),      \
        static_cast<int64_t>(shp.volume()), 0};                                \
                                                                               \
    memcpy(p, &desc, sizeof(CuDeviceArray));                                   \
    p += sizeof(CuDeviceArray);                                                \
  }

CUDA_DEVICE_ARRAY_ARG(read, read_accessor);    // cuda_device_array_arg_read
CUDA_DEVICE_ARRAY_ARG(write, write_accessor);  // cuda_device_array_arg_write

/* TODO find a better way to do this.
   Due to the templating for the accessor<T, DIM>, we cannot dispatch
   dynamically without some sort of lookup. We are using a switch case
   indirection. Probably can do some tricks with MACRO expansion
*/
template <typename T, int D>
void dispatch_access(AccessMode mode, char *&p,
                     const legate::PhysicalArray &arr) {
  if (mode == AccessMode::READ)
    cuda_device_array_arg_read<T, D>(p, arr);
  else
    cuda_device_array_arg_write<T, D>(p, arr);
}
template <typename T>
void dispatch_dim(AccessMode mode, int dim, char *&p,
                  const legate::PhysicalArray &arr) {
  switch (dim) {
    case 1:
      dispatch_access<T, 1>(mode, p, arr);
      break;
    case 2:
      dispatch_access<T, 2>(mode, p, arr);
      break;
    case 3:
      dispatch_access<T, 3>(mode, p, arr);
      break;
    default:
      throw std::runtime_error("Unsupported dimension");
  }
}
void dispatch_type(AccessMode mode, legate::Type::Code code, int dim, char *&p,
                   const legate::PhysicalArray &arr) {
  switch (code) {
    case legate::Type::Code::FLOAT32:
      dispatch_dim<float>(mode, dim, p, arr);
      break;
    case legate::Type::Code::FLOAT64:
      dispatch_dim<double>(mode, dim, p, arr);
      break;
    case legate::Type::Code::INT32:
      dispatch_dim<int32_t>(mode, dim, p, arr);
      break;
    case legate::Type::Code::INT64:
      dispatch_dim<int64_t>(mode, dim, p, arr);
      break;
    default:
      throw std::runtime_error("Unsupported element type");
  }
}

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

  const std::size_t padded_bytes = 16;

  // compute total size: all device arrays + all scalars
  // skip scalar 0-2 (kernel_name, threads, blocks)
  std::size_t buffer_size =
      padded_bytes + (num_inputs + num_outputs) * sizeof(CuDeviceArray);
  for (std::size_t i = ARG_OFFSET; i < num_scalars; ++i)
    buffer_size += context.scalar(i).size();

  std::vector<char> arg_buffer(buffer_size);
  char *p = arg_buffer.data() + padded_bytes;
  for (std::size_t i = 0; i < num_inputs; ++i) {
    auto ps = context.input(i);
    auto code = ps.type().code();
    auto dim = ps.dim();
    dispatch_type(ufi::AccessMode::READ, code, dim, p, ps);
  }
  for (std::size_t i = 0; i < num_outputs; ++i) {
    auto ps = context.output(i);
    auto code = ps.type().code();
    auto dim = ps.dim();
    dispatch_type(ufi::AccessMode::WRITE, code, dim, p, ps);
  }
  for (std::size_t i = ARG_OFFSET; i < num_scalars; ++i) {
    const auto &scalar = context.scalar(i);
    memcpy(p, scalar.ptr(), scalar.size());
    p += scalar.size();
  }

  void *config[] = {
      CU_LAUNCH_PARAM_BUFFER_POINTER,
      static_cast<void *>(arg_buffer.data()),
      CU_LAUNCH_PARAM_BUFFER_SIZE,
      &buffer_size,
      CU_LAUNCH_PARAM_END,
  };

  CUstream custream_ = reinterpret_cast<CUstream>(stream_);

  DRIVER_ERROR_CHECK(cuLaunchKernel(func, tx, ty, tz, bx, by, bz, 0, custream_,
                                    nullptr, config));

  // DRIVER_ERROR_CHECK(cuStreamSynchronize(stream_));
  // TEST_PRINT_DEBUG(a, N, float, "%f", stream_, "array a");
  // TEST_PRINT_DEBUG(b, N, float, "%f", stream_, "array b");
  // TEST_PRINT_DEBUG(c, N, float, "%f", stream_, "array c");
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

/* allignment contrainsts are transitive.
    we can allign all the inputs and then alligns all the outputs
    then allign one input with one output
    This reduces the need for a cartesian product.
*/
inline void add_transitive_alignment(
    legate::AutoTask &task, const std::vector<legate::Variable> &inputs,
    const std::vector<legate::Variable> &outputs) {
  for (size_t i = 1; i < inputs.size(); ++i)
    task.add_constraint(legate::align(inputs[i], inputs[0]));
  for (size_t i = 1; i < outputs.size(); ++i)
    task.add_constraint(legate::align(outputs[i], outputs[0]));
  if (!inputs.empty() && !outputs.empty())
    task.add_constraint(legate::align(outputs[0], inputs[0]));
}

legate::Library get_lib() {
  auto runtime = cupynumeric::CuPyNumericRuntime::get_runtime();
  return runtime->get_library();
}

inline void add_xyz_scalars(legate::AutoTask &task,
                            const std::vector<uint32_t> &v) {
  uint32_t xyz[3] = {1, 1, 1};
  const size_t n = std::min<size_t>(3, v.size());
  for (size_t i = 0; i < n; ++i) xyz[i] = v[i];

  task.add_scalar_arg(legate::Scalar(xyz[0]));
  task.add_scalar_arg(legate::Scalar(xyz[1]));
  task.add_scalar_arg(legate::Scalar(xyz[2]));
}

void new_task(std::string kernel_name, std::vector<uint32_t> &blocks,
              std::vector<uint32_t> &threads,
              std::vector<std::shared_ptr<CN_NDArray>> &inputs,
              std::vector<std::shared_ptr<CN_NDArray>> &outputs,
              std::vector<legate::Scalar> &scalars) {
  auto runtime = legate::Runtime::get_runtime();
  auto library = get_lib();
  auto task =
      runtime->create_task(library, legate::LocalTaskID{ufi::RUN_PTX_TASK});

  // Use first output shape as reference
  const auto &out_shape = outputs.front()->obj.shape();

  std::vector<legate::Variable> input_vars;
  std::vector<legate::Variable> output_vars;

  for (const auto &out_ptr : outputs) {
    cupynumeric::NDArray &out = out_ptr->obj;
    auto store = out.get_store();
    auto p = task.add_output(store);
    output_vars.push_back(p);
  }

  for (const auto &in_ptr : inputs) {
    cupynumeric::NDArray &in = in_ptr->obj;
    auto store = in.get_store();
    auto p = task.add_input(broadcast(out_shape, store));
    input_vars.push_back(p);
  }

  // Add kernel name and scalar args
  task.add_scalar_arg(legate::Scalar(kernel_name));  // 0
  add_xyz_scalars(task, blocks);                     // bx,by,bz 1,2,3
  add_xyz_scalars(task, threads);                    // tx,ty,tz 4,5,6

  for (const auto &scalar : scalars)
    task.add_scalar_arg(scalar);  // 7+ -> ARG_OFFSET

  /* TODO actually support the constraint system */
  // Add alignment constraints
  add_transitive_alignment(task, input_vars, output_vars);

  runtime->submit(std::move(task));
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