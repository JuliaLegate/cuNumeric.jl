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

#include <cupynumeric.h>
#include <cupynumeric/runtime.h>
#include <deps/realm/machine.h>
#include <deps/realm/machine_impl.h>
#include <legate.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>

#include "ndarray_c_api.h"

constexpr uint64_t KiB = 1024ull;
constexpr uint64_t MiB = KiB * 1024ull;
constexpr uint64_t GiB = MiB * 1024ull;

using Legion::Machine;

static inline uint64_t query_machine_config_common(
    Realm::Processor::Kind proc_kind, Realm::Memory::Kind mem_kind) {
  Machine legion_machine{Machine::get_machine()};
  uint64_t total_mem = 0;

  Machine::ProcessorQuery procs =
      Machine::ProcessorQuery(legion_machine).only_kind(proc_kind);

  for (auto pit = procs.begin(); pit != procs.end(); ++pit) {
    auto proc = *pit;
    assert(proc.kind() == proc_kind);

    Realm::Machine::MemoryQuery local_memories =
        Machine::MemoryQuery(legion_machine)
            .only_kind(mem_kind)
            .same_address_space_as(proc);

    for (auto mit = local_memories.begin(); mit != local_memories.end();
         ++mit) {
      auto mem = *mit;
      assert(mem.kind() == mem_kind);
      total_mem += mem.capacity();
    }
  }

  return total_mem;
}

extern "C" {

static inline uint64_t query_allocated_bytes_common(
    Realm::Processor::Kind proc_kind, Realm::Memory::Kind mem_kind) {
  Machine legion_machine{Machine::get_machine()};
  auto legion_runtime = Legion::Runtime::get_runtime();
  auto ctx = Legion::Runtime::get_context();

  uint64_t current_bytes = 0;

  Machine::ProcessorQuery procs =
      Machine::ProcessorQuery(legion_machine).only_kind(proc_kind);

  for (auto pit = procs.begin(); pit != procs.end(); ++pit) {
    auto proc = *pit;
    assert(proc.kind() == proc_kind);

    Realm::Machine::MemoryQuery local_memories =
        Machine::MemoryQuery(legion_machine)
            .only_kind(mem_kind)
            .same_address_space_as(proc);

    for (auto mit = local_memories.begin(); mit != local_memories.end();
         ++mit) {
      auto mem = *mit;
      assert(mem.kind() == mem_kind);

      size_t available = legion_runtime->query_available_memory(ctx, mem);
      size_t capacity = mem.capacity();
      current_bytes += (capacity - available);
    }
  }

  return current_bytes;
}

uint64_t nda_query_allocated_device_memory() {
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  uint64_t allocated = query_allocated_bytes_common(Realm::Processor::TOC_PROC,
                                                    Realm::Memory::GPU_FB_MEM);
#else
  uint64_t allocated = 0;
#endif
  return allocated;
}
uint64_t nda_query_allocated_host_memory() {
  return query_allocated_bytes_common(Realm::Processor::LOC_PROC,
                                      Realm::Memory::SYSTEM_MEM);
}

uint64_t nda_query_total_device_memory() {
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  uint64_t total = query_machine_config_common(Realm::Processor::TOC_PROC,
                                               Realm::Memory::GPU_FB_MEM);
#else
  uint64_t total = 0;
#endif
  return total;
}

uint64_t nda_query_total_host_memory() {
  return query_machine_config_common(Realm::Processor::LOC_PROC,
                                     Realm::Memory::SYSTEM_MEM);
}

}  // extern "C"