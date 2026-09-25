#pragma once

#include <cstdint>
#include <type_traits>

#include "legate.h"

namespace ufi {
// The wire values come from Legate. Julia receives this enum through CxxWrap
// and never defines a parallel table of integer operation codes.
enum class MapReduceOp : int32_t {
  ADD = static_cast<int32_t>(legate::ReductionOpKind::ADD),
  MUL = static_cast<int32_t>(legate::ReductionOpKind::MUL),
  MIN = static_cast<int32_t>(legate::ReductionOpKind::MIN),
  MAX = static_cast<int32_t>(legate::ReductionOpKind::MAX),
};
using MapReduceOpValue = std::underlying_type_t<MapReduceOp>;

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
legate::Library get_mapreduce_library();
void register_mapreduce_tasks();

class RunPTXMapReduceTask : public legate::LegateTask<RunPTXMapReduceTask> {
 public:
  static inline const auto TASK_CONFIG =
      legate::TaskConfig{legate::LocalTaskID{0}}.with_variant_options(
          legate::VariantOptions{}.with_has_allocations(true).with_elide_device_ctx_sync(true));
  static void gpu_variant(legate::TaskContext context);
};

class RunPTXReduceFinishTask : public legate::LegateTask<RunPTXReduceFinishTask> {
 public:
  static inline const auto TASK_CONFIG =
      legate::TaskConfig{legate::LocalTaskID{1}}.with_variant_options(
          legate::VariantOptions{}.with_elide_device_ctx_sync(true));
  static void gpu_variant(legate::TaskContext context);
};
#endif
}  // namespace ufi
