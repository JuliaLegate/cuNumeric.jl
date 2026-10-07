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

#include "types.h"
#include "cupynumeric.h"

#include <vector>

void wrap_unary_ops(jlcxx::Module& mod) {

  mod.add_enum<CuPyNumericUnaryOpCode>("UnaryOpCode",
    std::vector<const char*>(
        {"ABSOLUTE", "ANGLE", "ARCCOS", "ARCCOSH", "ARCSIN", "ARCSINH", "ARCTAN", "ARCTANH",
         "CBRT", "CEIL", "CLIP", "CONJ", "COPY", "COS", "COSH", "DEG2RAD",
         "EXP", "EXP2", "EXPM1", "FLOOR", "FREXP", "GETARG", "IMAG", "INVERT",
         "ISFINITE", "ISINF", "ISNAN", "LOG", "LOG10", "LOG1P", "LOG2",
         "LOGICAL_NOT", "MODF", "NEGATIVE", "POSITIVE", "RAD2DEG", "REAL",
         "RECIPROCAL", "RINT", "ROUND", "SIGN", "SIGNBIT", "SIN", "SINH",
         "SQRT", "SQUARE", "TAN", "TANH", "TRUNC"}),
    std::vector<int>({
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_ABSOLUTE),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_ANGLE),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_ARCCOS),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_ARCCOSH),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_ARCSIN),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_ARCSINH),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_ARCTAN),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_ARCTANH),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_CBRT),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_CEIL),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_CLIP),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_CONJ),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_COPY),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_COS),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_COSH),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_DEG2RAD),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_EXP),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_EXP2),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_EXPM1),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_FLOOR),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_FREXP),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_GETARG),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_IMAG),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_INVERT),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_ISFINITE),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_ISINF),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_ISNAN),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_LOG),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_LOG10),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_LOG1P),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_LOG2),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_LOGICAL_NOT),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_MODF),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_NEGATIVE),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_POSITIVE),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_RAD2DEG),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_REAL),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_RECIPROCAL),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_RINT),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_ROUND),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_SIGN),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_SIGNBIT),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_SIN),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_SINH),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_SQRT),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_SQUARE),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_TAN),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_TANH),
        static_cast<int>(CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_TRUNC),

    })
  );
}

void wrap_unary_reds(jlcxx::Module& mod) {

  mod.add_enum<CuPyNumericUnaryRedCode>("UnaryRedCode",
    std::vector<const char*>(
        {"ALL", "ANY", "ARGMAX", "ARGMIN", "CONTAINS", "COUNT_NONZERO", "MAX",
         "MIN", "NANARGMAX", "NANARGMIN", "NANMAX", "NANMIN", "NANPROD",
         "NANSUM", "PROD", "SUM", "SUM_SQUARES", "VARIANCE"}),
    std::vector<int>({
        static_cast<int>(CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_ALL),
        static_cast<int>(CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_ANY),
        static_cast<int>(CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_ARGMAX),
        static_cast<int>(CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_ARGMIN),
        static_cast<int>(CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_CONTAINS),
        static_cast<int>(CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_COUNT_NONZERO),
        static_cast<int>(CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_MAX),
        static_cast<int>(CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_MIN),
        static_cast<int>(CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_NANARGMAX),
        static_cast<int>(CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_NANARGMIN),
        static_cast<int>(CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_NANMAX),
        static_cast<int>(CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_NANMIN),
        static_cast<int>(CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_NANPROD),
        static_cast<int>(CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_NANSUM),
        static_cast<int>(CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_PROD),
        static_cast<int>(CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_SUM),
        static_cast<int>(CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_SUM_SQUARES),
        static_cast<int>(CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_VARIANCE)
    })
  );
}

void wrap_binary_ops(jlcxx::Module& mod) {

  mod.add_enum<CuPyNumericBinaryOpCode>("BinaryOpCode",
    std::vector<const char*>({
        "ADD", "ARCTAN2", "BITWISE_AND", "BITWISE_OR", "BITWISE_XOR",
        "COPYSIGN", "DIVIDE", "EQUAL", "FLOAT_POWER", "FLOOR_DIVIDE",
        "FMOD", "GCD", "GREATER", "GREATER_EQUAL", "HYPOT", "ISCLOSE", "LCM", "LDEXP", "LEFT_SHIFT",
        "LESS", "LESS_EQUAL", "LOGADDEXP", "LOGADDEXP2", "LOGICAL_AND", "LOGICAL_OR", "LOGICAL_XOR",
        "MAXIMUM", "MINIMUM", "MOD", "MULTIPLY", "NEXTAFTER",
        "NOT_EQUAL", "POWER", "RIGHT_SHIFT", "SUBTRACT"
    }),
    std::vector<int>({
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_ADD),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_ARCTAN2),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_BITWISE_AND),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_BITWISE_OR),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_BITWISE_XOR),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_COPYSIGN),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_DIVIDE),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_EQUAL),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_FLOAT_POWER),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_FLOOR_DIVIDE),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_FMOD),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_GCD),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_GREATER),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_GREATER_EQUAL),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_HYPOT),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_ISCLOSE),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_LCM),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_LDEXP),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_LEFT_SHIFT),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_LESS),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_LESS_EQUAL),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_LOGADDEXP),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_LOGADDEXP2),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_LOGICAL_AND),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_LOGICAL_OR),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_LOGICAL_XOR),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_MAXIMUM),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_MINIMUM),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_MOD),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_MULTIPLY),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_NEXTAFTER),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_NOT_EQUAL),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_POWER),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_RIGHT_SHIFT),
        static_cast<int>(CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_SUBTRACT)
    })    
  );  
}

void wrap_linalg_ops(jlcxx::Module& mod) {
  mod.set_const("SOLVE",
                legate::LocalTaskID{CuPyNumericOpCode::CUPYNUMERIC_SOLVE});
  mod.set_const("MP_SOLVE",
                legate::LocalTaskID{CuPyNumericOpCode::CUPYNUMERIC_MP_SOLVE});
  mod.set_const("MP_POTRF",
                legate::LocalTaskID{CuPyNumericOpCode::CUPYNUMERIC_MP_POTRF});
  mod.set_const("MP_QR",
                legate::LocalTaskID{CuPyNumericOpCode::CUPYNUMERIC_MP_QR});
  mod.set_const("POTRS",
                legate::LocalTaskID{CuPyNumericOpCode::CUPYNUMERIC_POTRS});
  mod.set_const("TRSM",
                legate::LocalTaskID{CuPyNumericOpCode::CUPYNUMERIC_TRSM});
  mod.set_const("SYRK",
                legate::LocalTaskID{CuPyNumericOpCode::CUPYNUMERIC_SYRK});
  mod.set_const("GEMM",
                legate::LocalTaskID{CuPyNumericOpCode::CUPYNUMERIC_GEMM});
  mod.set_const(
      "TRANSPOSE_COPY_2D",
      legate::LocalTaskID{CuPyNumericOpCode::CUPYNUMERIC_TRANSPOSE_COPY_2D});
  mod.set_const("TRILU",
                legate::LocalTaskID{CuPyNumericOpCode::CUPYNUMERIC_TRILU});
  mod.set_const("SVD", legate::LocalTaskID{CuPyNumericOpCode::CUPYNUMERIC_SVD});
  mod.set_const("CQR", legate::LocalTaskID{CuPyNumericOpCode::CUPYNUMERIC_QR});
  mod.set_const("POTRF",
                legate::LocalTaskID{CuPyNumericOpCode::CUPYNUMERIC_POTRF});
  mod.set_const("SYEV",
                legate::LocalTaskID{CuPyNumericOpCode::CUPYNUMERIC_SYEV});
  mod.set_const("GEEV",
                legate::LocalTaskID{CuPyNumericOpCode::CUPYNUMERIC_GEEV});
}

void wrap_fft_ops(jlcxx::Module& mod) {
  mod.set_const("FFT", legate::LocalTaskID{CuPyNumericOpCode::CUPYNUMERIC_FFT});
  mod.set_const("FFT_R2C", int32_t(CUPYNUMERIC_FFT_R2C));
  mod.set_const("FFT_C2R", int32_t(CUPYNUMERIC_FFT_C2R));
  mod.set_const("FFT_C2C", int32_t(CUPYNUMERIC_FFT_C2C));
  mod.set_const("FFT_D2Z", int32_t(CUPYNUMERIC_FFT_D2Z));
  mod.set_const("FFT_Z2D", int32_t(CUPYNUMERIC_FFT_Z2D));
  mod.set_const("FFT_Z2Z", int32_t(CUPYNUMERIC_FFT_Z2Z));
  mod.set_const("FFT_FORWARD", int32_t(CUPYNUMERIC_FFT_FORWARD));
  mod.set_const("FFT_INVERSE", int32_t(CUPYNUMERIC_FFT_INVERSE));
}

void wrap_bitgenerator_ops(jlcxx::Module& mod) {
  mod.set_const(
      "BITGENERATOR",
      legate::LocalTaskID{CuPyNumericOpCode::CUPYNUMERIC_BITGENERATOR});

  mod.set_const("BITGENOP_CREATE", int32_t(CUPYNUMERIC_BITGENOP_CREATE));
  mod.set_const("BITGENOP_DESTROY", int32_t(CUPYNUMERIC_BITGENOP_DESTROY));
  mod.set_const("BITGENOP_RAND_RAW", int32_t(CUPYNUMERIC_BITGENOP_RAND_RAW));
  mod.set_const("BITGENOP_DISTRIBUTION",
                int32_t(CUPYNUMERIC_BITGENOP_DISTRIBUTION));

  mod.set_const("BITGENTYPE_DEFAULT", uint32_t(CUPYNUMERIC_BITGENTYPE_DEFAULT));
  mod.set_const("BITGENTYPE_XORWOW", uint32_t(CUPYNUMERIC_BITGENTYPE_XORWOW));
  mod.set_const("BITGENTYPE_MRG32K3A",
                uint32_t(CUPYNUMERIC_BITGENTYPE_MRG32K3A));
  mod.set_const("BITGENTYPE_MTGP32", uint32_t(CUPYNUMERIC_BITGENTYPE_MTGP32));
  mod.set_const("BITGENTYPE_MT19937", uint32_t(CUPYNUMERIC_BITGENTYPE_MT19937));
  mod.set_const("BITGENTYPE_PHILOX4_32_10",
                uint32_t(CUPYNUMERIC_BITGENTYPE_PHILOX4_32_10));

  mod.set_const("BITGENDIST_INTEGERS_16",
                uint32_t(CUPYNUMERIC_BITGENDIST_INTEGERS_16));
  mod.set_const("BITGENDIST_INTEGERS_32",
                uint32_t(CUPYNUMERIC_BITGENDIST_INTEGERS_32));
  mod.set_const("BITGENDIST_INTEGERS_64",
                uint32_t(CUPYNUMERIC_BITGENDIST_INTEGERS_64));
  mod.set_const("BITGENDIST_UNIFORM_32",
                uint32_t(CUPYNUMERIC_BITGENDIST_UNIFORM_32));
  mod.set_const("BITGENDIST_UNIFORM_64",
                uint32_t(CUPYNUMERIC_BITGENDIST_UNIFORM_64));
  mod.set_const("BITGENDIST_NORMAL_32",
                uint32_t(CUPYNUMERIC_BITGENDIST_NORMAL_32));
  mod.set_const("BITGENDIST_NORMAL_64",
                uint32_t(CUPYNUMERIC_BITGENDIST_NORMAL_64));
  mod.set_const("BITGENDIST_EXPONENTIAL_32",
                uint32_t(CUPYNUMERIC_BITGENDIST_EXPONENTIAL_32));
  mod.set_const("BITGENDIST_EXPONENTIAL_64",
                uint32_t(CUPYNUMERIC_BITGENDIST_EXPONENTIAL_64));
}
