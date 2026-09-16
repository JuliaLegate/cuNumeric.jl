#include "mapreduce.h"
#include "ptx.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "legate/data/buffer.h"
#include "legate/mapping/mapping.h"
#include "legate/redop/redop.h"

extern std::size_t padded_bytes_kernel_state;

namespace ufi {
namespace {
constexpr int THREADS = 256;
constexpr int64_t MAX_PARTIALS = 4096;

// These tasks need their own mapper: cuPyNumeric's mapper cannot account for
// scratch allocated by tasks registered outside its built-in task table.
class MapReduceMapper : public legate::mapping::Mapper {
 public:
  std::vector<legate::mapping::StoreMapping> store_mappings(
      const legate::mapping::Task&,
      const std::vector<legate::mapping::StoreTarget>&) override { return {}; }

  std::optional<std::size_t> allocation_pool_size(
      const legate::mapping::Task&, legate::mapping::StoreTarget target) override {
    // One partial buffer; ComplexF64 is the largest supported accumulator.
    return target == legate::mapping::StoreTarget::FBMEM
        ? MAX_PARTIALS * sizeof(legate::type_of<legate::Type::Code::COMPLEX128>) : 0;
  }

  legate::Scalar tunable_value(legate::TunableID) override {
    throw std::invalid_argument("mapreduce has no tunables");
  }
};

template <int D>
using ArrayArg = CuStridedDeviceArray<D>;

template <legate::Type::Code C>
constexpr bool supported = C == legate::Type::Code::BOOL ||
    C == legate::Type::Code::INT8 || C == legate::Type::Code::INT16 ||
    C == legate::Type::Code::INT32 || C == legate::Type::Code::INT64 ||
    C == legate::Type::Code::UINT8 || C == legate::Type::Code::UINT16 ||
    C == legate::Type::Code::UINT32 || C == legate::Type::Code::UINT64 ||
    C == legate::Type::Code::FLOAT32 || C == legate::Type::Code::FLOAT64 ||
    C == legate::Type::Code::COMPLEX64 || C == legate::Type::Code::COMPLEX128;

template <int D, bool WRITE>
struct PackArray {
  template <legate::Type::Code C>
  ArrayArg<D> operator()(const legate::PhysicalStore& store) const {
    if constexpr (supported<C>) {
      using T = legate::type_of<C>;
      auto rect = store.shape<D>();
      auto acc = [&]() {
        if constexpr (WRITE) return store.write_accessor<T, D>();
        else return store.read_accessor<T, D>();
      }();
      ArrayArg<D> arg{};
      arg.ptr = const_cast<void*>(static_cast<const void*>(acc.ptr(rect.lo)));
      arg.length = rect.volume();
      arg.maxsize = arg.length * sizeof(T);
      for (int d = 0; d < D; ++d) {
        arg.dims[d] = rect.hi[d] - rect.lo[d] + 1;
        arg.strides[d] = acc.accessor.strides[d] / sizeof(T);
      }
      return arg;
    } else {
      throw std::invalid_argument("mapreduce: unsupported physical element type");
    }
  }
};

static void launch(CUfunction kernel, int blocks, cudaStream_t stream,
                   std::vector<void*>& args) {
  auto status = cuLaunchKernel(kernel, blocks, 1, 1, THREADS, 1, 1, 0,
                               reinterpret_cast<CUstream>(stream), args.data(), nullptr);
  if (status != CUDA_SUCCESS) {
    const char* message = nullptr;
    cuGetErrorString(status, &message);
    throw std::runtime_error(std::string("mapreduce PTX launch: ") + (message ? message : "unknown CUDA error"));
  }
}

template <class OP, int D>
ArrayArg<D> reduction_arg(const legate::PhysicalStore& store,
                          const legate::Rect<D>& bounds, uint64_t axes, bool single) {
  using T = typename OP::RHS;
  size_t strides[D];
  ArrayArg<D> arg{};
  arg.ptr = single ? store.write_accessor<T, D>().ptr(bounds, strides)
                   : store.reduce_accessor<OP, true, D>().ptr(bounds, strides);
  arg.length = 1;
  for (int d = 0; d < D; ++d) {
    // Promoted axes alias the same destination. Exclude those coordinates
    // so exactly one Julia thread updates each physical output element.
    arg.dims[d] = ((axes >> d) & 1) ? 1 : bounds.hi[d] - bounds.lo[d] + 1;
    arg.strides[d] = strides[d]; // ptr(rect, strides) returns element strides
    arg.length *= arg.dims[d];
  }
  arg.maxsize = arg.length * sizeof(T);
  return arg;
}

template <int D>
void contribute(CUfunction kernel, cudaStream_t stream, void* state,
                 ArrayArg<1>& scratch, ArrayArg<D>& dest,
                 bool single, int64_t start, int64_t count, int64_t chunks) {
  std::vector<void*> args{state, &scratch, &dest, &single, &start, &count, &chunks};
  launch(kernel, static_cast<int>((count + THREADS - 1) / THREADS), stream, args);
}

template <class OP, int D>
void run_reduction(legate::TaskContext& context, CUfunction kernel) {
  using T = typename OP::RHS;
  auto input = context.input(0).data();
  auto single = context.scalar(6).value<bool>();
  auto red = single ? context.output(0).data() : context.reduction(0).data();
  const auto rect = input.shape<D>();
  if (rect.empty()) return;
  auto stream = context.get_task_stream();
  auto axes = context.scalar(1).value<uint64_t>();
  auto full = context.scalar(2).value<bool>();
  auto combine = lookup_ptx(context.scalar(5).value<std::string>(), stream);
  auto src = legate::type_dispatch(input.type().code(), PackArray<D, false>{}, input);
  int64_t reduced = 1, retained = 1;
  for (int d = 0; d < D; ++d)
    (((axes >> d) & 1) ? reduced : retained) *= src.dims[d];

  // Deferred buffers belong to the task; kernels may still be queued when
  // their C++ handles leave scope. Reuse is ordered on the task's stream.
  auto buffer = legate::create_buffer<T>(MAX_PARTIALS, legate::Memory::Kind::GPU_FB_MEM);
  ArrayArg<1> scratch{buffer.ptr(0), MAX_PARTIALS * int64_t(sizeof(T)),
                      {MAX_PARTIALS}, {1}, MAX_PARTIALS};
  std::vector<uint8_t> state(padded_bytes_kernel_state, 0);
  for (int64_t start = 0; start < retained;) {
    int64_t count = std::min(MAX_PARTIALS, retained - start);
    int64_t chunks = std::min(MAX_PARTIALS / count, 1 + (reduced - 1) / 1024);
    std::vector<void*> args{state.data(), &src, &scratch};
    // Zero-size Julia singleton arguments are absent from the PTX signature.
    if (context.scalar(4).size()) args.push_back(const_cast<void*>(context.scalar(4).ptr()));
    args.push_back(&start);
    args.push_back(&chunks);
    launch(kernel, static_cast<int>(count * chunks), stream, args);
    if (full) {
      auto dest = reduction_arg<OP, 1>(red, red.shape<1>(), 1, single);
      contribute(combine, stream, state.data(), scratch, dest, single, start, count, chunks);
    } else {
      auto dest = reduction_arg<OP, D>(red, rect, axes, single);
      contribute(combine, stream, state.data(), scratch, dest, single, start, count, chunks);
    }
    start += count;
  }
}

struct ReduceDispatch {
  template <legate::Type::Code C, int D>
  void operator()(legate::TaskContext& ctx, CUfunction kernel) const {
    if constexpr (supported<C> && C != legate::Type::Code::BOOL) {
      using T = legate::type_of<C>;
      auto op = static_cast<MapReduceOp>(ctx.scalar(3).value<MapReduceOpValue>());
      if (op == MapReduceOp::ADD) return run_reduction<legate::SumReduction<T>, D>(ctx, kernel);
      if constexpr (C != legate::Type::Code::COMPLEX128) {
        if (op == MapReduceOp::MUL) return run_reduction<legate::ProdReduction<T>, D>(ctx, kernel);
      }
      if constexpr (C != legate::Type::Code::COMPLEX64 && C != legate::Type::Code::COMPLEX128) {
        if (op == MapReduceOp::MIN) return run_reduction<legate::MinReduction<T>, D>(ctx, kernel);
        if (op == MapReduceOp::MAX) return run_reduction<legate::MaxReduction<T>, D>(ctx, kernel);
      }
    }
    throw std::invalid_argument("mapreduce: unsupported reduction/type combination");
  }
};

struct FinishDispatch {
  template <int D>
  void operator()(legate::TaskContext& ctx, CUfunction kernel) const {
    auto input = ctx.input(0).data();
    auto output = ctx.output(0).data();
    if (output.shape<D>().empty()) return;
    auto src = legate::type_dispatch(input.type().code(), PackArray<D, false>{}, input);
    auto dst = legate::type_dispatch(output.type().code(), PackArray<D, true>{}, output);
    std::vector<uint8_t> state(padded_bytes_kernel_state, 0);
    std::vector<void*> args{state.data(), &src, &dst};
    if (ctx.scalar(1).size()) args.push_back(const_cast<void*>(ctx.scalar(1).ptr()));
    auto blocks = static_cast<int>(std::min(MAX_PARTIALS, 1 + (dst.length - 1) / THREADS));
    launch(kernel, blocks, ctx.get_task_stream(), args);
  }
};
}  // namespace

legate::Library get_mapreduce_library() {
  return legate::Runtime::get_runtime()->find_library("cuNumeric_mapreduce");
}

void register_mapreduce_tasks() {
  auto library = legate::Runtime::get_runtime()->create_library(
      "cuNumeric_mapreduce", legate::ResourceConfig{}, std::make_unique<MapReduceMapper>());
  RunPTXMapReduceTask::register_variants(library);
  RunPTXReduceFinishTask::register_variants(library);
}

void RunPTXMapReduceTask::gpu_variant(legate::TaskContext context) {
  auto kernel = lookup_ptx(context.scalar(0).value<std::string>(), context.get_task_stream());
  auto dest = context.scalar(6).value<bool>() ? context.output(0) : context.reduction(0);
  legate::double_dispatch(context.input(0).dim(), dest.type().code(),
                          ReduceDispatch{}, context, kernel);
}

void RunPTXReduceFinishTask::gpu_variant(legate::TaskContext context) {
  auto kernel = lookup_ptx(context.scalar(0).value<std::string>(), context.get_task_stream());
  legate::dim_dispatch(context.output(0).dim(), FinishDispatch{}, context, kernel);
}
}  // namespace ufi
