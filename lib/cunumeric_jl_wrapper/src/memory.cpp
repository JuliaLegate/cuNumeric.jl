#include <cupynumeric.h>
#include <cupynumeric/runtime.h>
#include <deps/realm/machine.h>
#include <deps/realm/machine_impl.h>
#include <legate.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>

constexpr uint64_t KiB = 1024ull;
constexpr uint64_t MiB = KiB * 1024ull;
constexpr uint64_t GiB = MiB * 1024ull;

using Legion::Machine;
static uint64_t query_machine_config() {
  Machine legion_machine{Machine::get_machine()};

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  Machine::ProcessorQuery gpus = Machine::ProcessorQuery(legion_machine)
                                     .only_kind(Realm::Processor::TOC_PROC);

  uint64_t total_fb_mem = 0;
  uint64_t gpus_count = gpus.count();

  // for each GPU
  for (auto it = gpus.begin(); it != gpus.end(); ++it) {
    auto proc = *it;
    assert(proc.kind() == Realm::Processor::TOC_PROC);

    // get all the FB memories local to this GPU
    Realm::Machine::MemoryQuery local_memories =
        Machine::MemoryQuery(legion_machine)
            .only_kind(Realm::Memory::GPU_FB_MEM)
            .same_address_space_as(proc);

    // TODO: will this ever have multiple GPU_FB_MEM memories???
    for (auto mem_it = local_memories.begin(); mem_it != local_memories.end();
         ++mem_it) {
      auto mem = *mem_it;
      assert(mem.kind() == Realm::Memory::GPU_FB_MEM);

      total_fb_mem += mem.capacity();
    }
  }
  // std::cout << "Detected " << gpus_count << " GPUs with " << total_fb_mem /
  // MiB << " MB each, total " << total_fb_mem / GiB << " GB\n";
  return total_fb_mem;
#else
  uint64_t total_system_mem = 0;
  Machine::ProcessorQuery cpus = Machine::ProcessorQuery(legion_machine)
                                     .only_kind(Realm::Processor::LOC_PROC);

  for (auto it = cpus.begin(); it != cpus.end(); ++it) {
    auto proc = *it;
    assert(proc.kind() == Realm::Processor::LOC_PROC);

    // get all the SYSTEM memories local to this CPU
    Realm::Machine::MemoryQuery local_memories =
        Machine::MemoryQuery(legion_machine)
            .only_kind(Realm::Memory::SYSTEM_MEM)
            .same_address_space_as(proc);

    // TODO: will this ever have multiple SYSTEM memories???
    for (auto mem_it = local_memories.begin(); mem_it != local_memories.end();
         ++mem_it) {
      auto mem = *mem_it;
      assert(mem.kind() == Realm::Memory::SYSTEM_MEM);

      total_system_mem += mem.capacity();
    }
  }
  // std::cout << "System memory: " << total_system_mem / GiB << "GB\n";
  return total_system_mem;
#endif
}

int nda_recalibrate_allocator() {
  auto runtime = legate::Runtime::get_runtime();
  runtime->issue_execution_fence(true);  // block = true

  Machine legion_machine{Machine::get_machine()};
  auto legion_runtime = Legion::Runtime::get_runtime();
  auto ctx = Legion::Runtime::get_context();

  // query current allocated bytes
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  Machine::ProcessorQuery gpus = Machine::ProcessorQuery(legion_machine)
                                     .only_kind(Realm::Processor::TOC_PROC);

  uint64_t current_bytes = 0;

  // for each GPU
  for (auto it = gpus.begin(); it != gpus.end(); ++it) {
    auto proc = *it;
    assert(proc.kind() == Realm::Processor::TOC_PROC);

    // get all the FB memories local to this GPU
    Realm::Machine::MemoryQuery local_memories =
        Machine::MemoryQuery(legion_machine)
            .only_kind(Realm::Memory::GPU_FB_MEM)
            .same_address_space_as(proc);

    for (auto mem_it = local_memories.begin(); mem_it != local_memories.end();
         ++mem_it) {
      auto mem = *mem_it;
      assert(mem.kind() == Realm::Memory::GPU_FB_MEM);

      size_t available = legion_runtime->query_available_memory(ctx, mem);
      size_t capacity = mem.capacity();
      current_bytes += (capacity - available);
    }
  }
#else
  Machine::ProcessorQuery cpus = Machine::ProcessorQuery(legion_machine)
                                     .only_kind(Realm::Processor::LOC_PROC);

  uint64_t current_bytes = 0;

  // for each CPU
  for (auto it = cpus.begin(); it != cpus.end(); ++it) {
    auto proc = *it;
    assert(proc.kind() == Realm::Processor::LOC_PROC);

    // get all the SYSTEM memories local to this CPU
    Realm::Machine::MemoryQuery local_memories =
        Machine::MemoryQuery(legion_machine)
            .only_kind(Realm::Memory::SYSTEM_MEM)
            .same_address_space_as(proc);

    for (auto mem_it = local_memories.begin(); mem_it != local_memories.end();
         ++mem_it) {
      auto mem = *mem_it;
      assert(mem.kind() == Realm::Memory::SYSTEM_MEM);

      size_t available = legion_runtime->query_available_memory(ctx, mem);
      size_t capacity = mem.capacity();
      current_bytes += (capacity - available);
    }
  }
#endif

  return current_bytes;
}

uint64_t nda_query_device_memory() {
  uint64_t total = query_machine_config();
  if (total == 0) {
    total = 8ull * GiB;
    std::cerr << "Warning: Unable to query device memory, defaulting to 8 GB\n";
  }
  return total;
}