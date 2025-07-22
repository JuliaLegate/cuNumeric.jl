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

#pragma once

#include "jlcxx/jlcxx.hpp"
#include "legate.h"

namespace ufi {
enum TaskIDs {
  LOAD_PTX_TASK = 143432,
  RUN_PTX_TASK = 143433,
};

class LoadPTXTask : public legate::LegateTask<LoadPTXTask> {
 public:
  static inline const auto TASK_CONFIG =
      legate::TaskConfig{legate::LocalTaskID{ufi::LOAD_PTX_TASK}};

  static void gpu_variant(legate::TaskContext context);
};

class RunPTXTask : public legate::LegateTask<RunPTXTask> {
 public:
  static inline const auto TASK_CONFIG =
      legate::TaskConfig{legate::LocalTaskID{ufi::RUN_PTX_TASK}};

  static void gpu_variant(legate::TaskContext context);
};

}  // namespace ufi
void wrap_cuda_methods(jlcxx::Module& mod);