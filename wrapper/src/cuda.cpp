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

#include <regex>

#include "ufi.h"
#include "cupynumeric.h"
#include "legate.h"
#include "legate/utilities/proc_local_storage.h"
#include "legion.h"
#include "cuda.h"

#define ERROR_CHECK(x) { \
    cudaError_t status = x; \
    if (status != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
        cudaStreamDestroy(stream_); \
        exit(-1); \
    } \
}

#define DRIVER_ERROR_CHECK(x) { \
    CUresult status = x; \
    if (status != CUDA_SUCCESS) { \
        const char* err_str = nullptr; \
        cuGetErrorString(status, &err_str); \
        fprintf(stderr, "CUDA Driver Error at %s:%d: %s\n", __FILE__, __LINE__, err_str); \
        cudaStreamDestroy(stream_); \
        exit(-1); \
    } \
}

#define TEST_PRINT_DEBUG(dev_ptr, N, T, format, stream, message)         \
{                                                                        \
    std::vector<T> host_arr(N);                                          \
    ERROR_CHECK(cudaMemcpy(host_arr.data(),                              \
                           reinterpret_cast<const T*>(dev_ptr),          \
                           sizeof(T) * N,                                \
                           cudaMemcpyDeviceToHost));                     \
    ERROR_CHECK(cudaStreamSynchronize(stream));                          \
    fprintf(stderr, "[TEST_PRINT] %s: " format "\n", message, host_arr[0]); \
}

namespace ufi {
    using namespace Legion;
    static legate::ProcLocalStorage<uint64_t> cufunction_ptr{};

// https://github.com/nv-legate/legate.pandas/blob/branch-22.01/src/udf/eval_udf_gpu.cc
/*static*/ void RunPTXTask::gpu_variant(legate::TaskContext context)
{
  cudaStream_t stream_ = context.get_task_stream();

  CUfunction func = reinterpret_cast<CUfunction>(cufunction_ptr.get());
  int32_t N = context.scalar(0).value<int32_t>();

  void *a = (void*)context.input(0).data().read_accessor<float, 1>().ptr(Realm::Point<1>(0));
  void *b = (void*)context.input(1).data().read_accessor<float, 1>().ptr(Realm::Point<1>(0));
  void *c = context.output(0).data().write_accessor<float, 1>().ptr(Realm::Point<1>(0));

  unsigned int blockDimX = 256;
  unsigned int gridDimX = (N + blockDimX - 1) / blockDimX;

  void* args[] = { &a, &b, &c, &N }; 

  TEST_PRINT_DEBUG(a, N, float, "%f", stream_, "array a");
  TEST_PRINT_DEBUG(b, N, float, "%f", stream_, "array b");   
  TEST_PRINT_DEBUG(c, N, float, "%f", stream_, "array c");   

  fprintf(stderr, "N is  %u\n", N);
  fprintf(stderr, "running function :%p\n", func);

  DRIVER_ERROR_CHECK(cuLaunchKernel(
    func, gridDimX, 1, 1, blockDimX, 1, 1, 0, stream_, args, nullptr
  ));

  DRIVER_ERROR_CHECK(cuStreamSynchronize(stream_));
  TEST_PRINT_DEBUG(c, N, float, "%f", stream_, "after array c");   
}



// https://github.com/nv-legate/legate.pandas/blob/branch-22.01/src/udf/load_ptx.cc
/*static*/ void LoadPTXTask::gpu_variant(legate::TaskContext context)
{
    std::string ptx = context.scalar(0).value<std::string>();

    std::cerr << ptx << std::endl;

    // auto output   = context.output(0);
    // auto output_acc = output.data().write_accessor<uint64_t, 1>();

    const unsigned num_options = 4;
    const size_t buffer_size   = 16384;
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
    CUresult result = cuModuleLoadDataEx(&module, static_cast<const void*>(ptx.c_str()), num_options, jit_options, option_vals);
    if (result != CUDA_SUCCESS) {
        if (result == CUDA_ERROR_OPERATING_SYSTEM) {
        fprintf(stderr,
                "ERROR: Device side asserts are not supported by the "
                "CUDA driver for MAC OSX, see NVBugs 1628896.\n");
        exit(-1);
        } else if (result == CUDA_ERROR_NO_BINARY_FOR_GPU) {
        fprintf(stderr, "ERROR: The binary was compiled for the wrong GPU architecture.\n");
        exit(-1);
        } else {
        fprintf(stderr, "Failed to load CUDA module! Error log: %s\n", log_error_buffer.data());
    #if CUDA_VERSION >= 6050
        const char *name, *str;
        assert(cuGetErrorName(result, &name) == CUDA_SUCCESS);
        assert(cuGetErrorString(result, &str) == CUDA_SUCCESS);
        fprintf(stderr, "CU: cuModuleLoadDataEx = %d (%s): %s\n", result, name, str);
    #else
        fprintf(stderr, "CU: cuModuleLoadDataEx = %d\n", result);
    #endif
        exit(-1);
        }
    }

    std::cmatch line_match;
    // there should be a built in find name of ufi function - pat
    bool match = std::regex_search(ptx.c_str(), line_match, std::regex(".visible .entry [_a-zA-Z0-9$]+"));

    const auto &matched_line = line_match.begin()->str();
    auto fun_name            = matched_line.substr(matched_line.rfind(" ") + 1, matched_line.size());

    CUfunction hfunc;
    result = cuModuleGetFunction(&hfunc, module, fun_name.c_str());
    assert(result == CUDA_SUCCESS);

    cufunction_ptr.emplace(reinterpret_cast<uint64_t>(hfunc));
    fprintf(stderr, "placed function :%p\n", hfunc);

    }
} // end ufi

// https://github.com/nv-legate/cupynumeric/blob/7e554b576ccc2d07a86986949cea79e56c690fe1/src/cupynumeric/ndarray.cc#L2120
// Copied method from the above link.
legate::LogicalStore broadcast(const std::vector<uint64_t>& shape,
                                        legate::LogicalStore& store) 
{
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

cupynumeric::NDArray new_task(legate::LogicalStore cufunc, cupynumeric::NDArray rhs1, cupynumeric::NDArray rhs2, cupynumeric::NDArray output, int32_t N) {
    auto runtime = legate::Runtime::get_runtime();
    auto library = get_lib();
    auto task = runtime->create_task(library, legate::LocalTaskID{ufi::RUN_PTX_TASK});
    // task.add_input(cufunc); // first input is the cufunction pointer

    auto& out_shape = output.shape();
    auto rhs1_temp = rhs1.get_store();
    auto rhs2_temp = rhs2.get_store();
    
    auto p_lhs  = task.add_output(output.get_store());
    auto p_rhs1 = task.add_input(broadcast(out_shape, rhs1_temp));
    auto p_rhs2 = task.add_input(broadcast(out_shape, rhs2_temp));

    task.add_scalar_arg(legate::Scalar(N));
    task.add_constraint(legate::align(p_lhs, p_rhs1));
    task.add_constraint(legate::align(p_rhs1, p_rhs2));

    runtime->submit(std::move(task));
    return output;
}


legate::LogicalStore ptx_task(std::string ptx) {
    auto runtime = legate::Runtime::get_runtime();
    auto library = get_lib();
    auto task = runtime->create_task(library, legate::LocalTaskID{ufi::LOAD_PTX_TASK});
    task.add_scalar_arg(legate::Scalar(ptx));

    auto scalar_store  = runtime->create_store({1}, legate::uint64(), false);
    task.add_output(scalar_store);

    runtime->submit(std::move(task));
    return scalar_store;
}



void register_tasks() {
    auto library = get_lib();
    ufi::LoadPTXTask::register_variants(library);
    ufi::RunPTXTask::register_variants(library);
}

void gpu_sync() {cudaDeviceSynchronize();}

void wrap_cuda_methods(jlcxx::Module& mod){
    mod.method("register_tasks", &register_tasks);
    mod.method("get_library", &get_lib);
    mod.method("new_task", &new_task);
    mod.method("ptx_task", &ptx_task);
    mod.method("gpu_sync", &gpu_sync);
}