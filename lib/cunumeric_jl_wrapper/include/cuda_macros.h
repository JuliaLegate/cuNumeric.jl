#pragma once

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
