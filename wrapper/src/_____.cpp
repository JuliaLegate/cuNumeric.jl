#include <cuda.h>
#include <legate.h>

#include <iostream>
#include <vector>

using namespace legate;

// You must define your task ID somewhere (pick unique number)
enum TaskIDs {
  MY_TASK_ID = 1000,
};

// Global CUDA module and function
CUmodule cuModule = nullptr;
CUfunction cuFunction = nullptr;

void init_cuda_module() {
  static bool initialized = false;
  if (initialized) return;
  initialized = true;

  cuInit(0);

  CUcontext context;
  cuCtxGetCurrent(&context);

  CUresult res = cuModuleLoad(&cuModule, "kernel_add.cubin");
  if (res != CUDA_SUCCESS) {
    std::cerr << "Failed to load CUDA module!" << std::endl;
    exit(1);
  }

  res = cuModuleGetFunction(&cuFunction, cuModule, "kernel_add");
  if (res != CUDA_SUCCESS) {
    std::cerr << "Failed to get CUDA function!" << std::endl;
    exit(1);
  }
}

void my_gpu_task_launcher(const Legion::Task* task,
                          const std::vector<Legion::PhysicalRegion>& regions,
                          Legion::Context ctx, Legion::Runtime* runtime) {
  init_cuda_module();

  // Extract device pointers from regions (assuming 3 input/output arrays)
  CUdeviceptr a_ptr = reinterpret_cast<CUdeviceptr>(
      regions[0].get_field_accessor(FID_VAL, LEGION_READ_ONLY).ptr(0));
  CUdeviceptr b_ptr = reinterpret_cast<CUdeviceptr>(
      regions[1].get_field_accessor(FID_VAL, LEGION_READ_ONLY).ptr(0));
  CUdeviceptr c_ptr = reinterpret_cast<CUdeviceptr>(
      regions[2].get_field_accessor(FID_VAL, LEGION_WRITE_ONLY).ptr(0));

  // TODO: figure out data size, e.g. from task->index_domain or other metadata
  int N = 1024;  // You should replace this with real data size

  int threads = 256;
  int blocks = (N + threads - 1) / threads;

  // Kernel args for cuLaunchKernel
  void* args[] = {&a_ptr, &b_ptr, &c_ptr};

  CUresult res = cuLaunchKernel(cuFunction, blocks, 1, 1, threads, 1, 1, 0,
                                nullptr, args, nullptr);
  if (res != CUDA_SUCCESS) {
    std::cerr << "CUDA kernel launch failed!" << std::endl;
    exit(1);
  }

  cuCtxSynchronize();
}

extern "C" void register_tasks() {
  TaskRegistrar registrar(MY_TASK_ID, "MyTask");
  registrar.add_variant(my_gpu_task_launcher, LEGATE_GPU_VARIANT);
}
