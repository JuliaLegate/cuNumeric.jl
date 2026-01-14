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

namespace legate_util {
template <legate::Type::Code CODE>
struct code_to_cxx;

#define DEFINE_CODE_TO_CXX(code_enum, cxx_type)       \
  template <>                                         \
  struct code_to_cxx<legate::Type::Code::code_enum> { \
    using type = cxx_type;                            \
  };

DEFINE_CODE_TO_CXX(BOOL, bool)
DEFINE_CODE_TO_CXX(INT8, int8_t)
DEFINE_CODE_TO_CXX(INT16, int16_t)
DEFINE_CODE_TO_CXX(INT32, int32_t)
DEFINE_CODE_TO_CXX(INT64, int64_t)
DEFINE_CODE_TO_CXX(UINT8, uint8_t)
DEFINE_CODE_TO_CXX(UINT16, uint16_t)
DEFINE_CODE_TO_CXX(UINT32, uint32_t)
DEFINE_CODE_TO_CXX(UINT64, uint64_t)
#ifdef HAVE_CUDA
DEFINE_CODE_TO_CXX(FLOAT16, __half)
#else
// Dummy type for FLOAT16 when CUDA is not available
// This allows compilation but we throw an error if actually used
struct __half_dummy {};
DEFINE_CODE_TO_CXX(FLOAT16, __half_dummy)
#endif
DEFINE_CODE_TO_CXX(FLOAT32, float)
DEFINE_CODE_TO_CXX(FLOAT64, double)
DEFINE_CODE_TO_CXX(COMPLEX64, std::complex<float>)
DEFINE_CODE_TO_CXX(COMPLEX128, std::complex<double>)
#undef DEFINE_CODE_TO_CXX
}  // namespace legate_util

// Unary op codes
void wrap_unary_ops(jlcxx::Module&);

// Unary reduction op codes
void wrap_unary_reds(jlcxx::Module&);

// Binary op codes
void wrap_binary_ops(jlcxx::Module&);
