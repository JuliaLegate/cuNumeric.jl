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

#include "types.h"

#include "cupynumeric.h"

void wrap_unary_ops(jlcxx::Module& mod) {
  mod.add_bits<CuPyNumericUnaryOpCode>("UnaryOpCode",
                                       jlcxx::julia_type("CppEnum"));
  mod.set_const("ABSOLUTE", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_ABSOLUTE);
  mod.set_const("ANGLE", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_ANGLE);
  mod.set_const("ARCCOS", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_ARCCOS);
  mod.set_const("ARCCOSH", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_ARCCOSH);
  mod.set_const("ARCSIN", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_ARCSIN);
  mod.set_const("ARCSINH", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_ARCSINH);
  mod.set_const("ARCTAN", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_ARCTAN);
  mod.set_const("ARCTANH", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_ARCTANH);
  mod.set_const("CBRT", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_CBRT);
  mod.set_const("CEIL", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_CEIL);
  mod.set_const("CLIP", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_CLIP);
  mod.set_const("CONJ", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_CONJ);
  mod.set_const("COPY", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_COPY);
  mod.set_const("COS", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_COS);
  mod.set_const("COSH", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_COSH);
  mod.set_const("DEG2RAD", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_DEG2RAD);
  mod.set_const("EXP", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_EXP);
  mod.set_const("EXP2", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_EXP2);
  mod.set_const("EXPM1", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_EXPM1);
  mod.set_const("FLOOR", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_FLOOR);
  mod.set_const("FREXP", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_FREXP);
  mod.set_const("GETARG", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_GETARG);
  mod.set_const("IMAG", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_IMAG);
  mod.set_const("INVERT", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_INVERT);
  mod.set_const("ISFINITE", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_ISFINITE);
  mod.set_const("ISINF", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_ISINF);
  mod.set_const("ISNAN", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_ISNAN);
  mod.set_const("LOG", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_LOG);
  mod.set_const("LOG10", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_LOG10);
  mod.set_const("LOG1P", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_LOG1P);
  mod.set_const("LOG2", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_LOG2);
  mod.set_const("LOGICAL_NOT",
                CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_LOGICAL_NOT);
  mod.set_const("MODF", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_MODF);
  mod.set_const("NEGATIVE", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_NEGATIVE);
  mod.set_const("POSITIVE", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_POSITIVE);
  mod.set_const("RAD2DEG", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_RAD2DEG);
  mod.set_const("REAL", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_REAL);
  mod.set_const("RECIPROCAL",
                CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_RECIPROCAL);
  mod.set_const("RINT", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_RINT);
  mod.set_const("ROUND", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_ROUND);
  mod.set_const("SIGN", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_SIGN);
  mod.set_const("SIGNBIT", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_SIGNBIT);
  mod.set_const("SIN", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_SIN);
  mod.set_const("SINH", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_SINH);
  mod.set_const("SQRT", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_SQRT);
  mod.set_const("SQUARE", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_SQUARE);
  mod.set_const("TAN", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_TAN);
  mod.set_const("TANH", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_TANH);
  mod.set_const("TRUNC", CuPyNumericUnaryOpCode::CUPYNUMERIC_UOP_TRUNC);
}

void wrap_unary_reds(jlcxx::Module& mod) {
  mod.add_bits<CuPyNumericUnaryRedCode>("UnaryRedCode",
                                        jlcxx::julia_type("CppEnum"));
  mod.set_const("ALL", CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_ALL);
  mod.set_const("ANY", CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_ANY);
  mod.set_const("ARGMAX", CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_ARGMAX);
  mod.set_const("ARGMIN", CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_ARGMIN);
  mod.set_const("CONTAINS", CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_CONTAINS);
  mod.set_const("COUNT_NONZERO",
                CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_COUNT_NONZERO);
  mod.set_const("MAX", CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_MAX);
  mod.set_const("MIN", CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_MIN);
  mod.set_const("NANARGMAX",
                CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_NANARGMAX);
  mod.set_const("NANARGMIN",
                CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_NANARGMIN);
  mod.set_const("NANMAX", CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_NANMAX);
  mod.set_const("NANMIN", CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_NANMIN);
  mod.set_const("NANPROD", CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_NANPROD);
  mod.set_const("NANSUM", CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_NANSUM);
  mod.set_const("PROD", CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_PROD);
  mod.set_const("SUM", CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_SUM);
  mod.set_const("SUM_SQUARES",
                CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_SUM_SQUARES);
  mod.set_const("VARIANCE", CuPyNumericUnaryRedCode::CUPYNUMERIC_RED_VARIANCE);
}

void wrap_binary_ops(jlcxx::Module& mod) {
  mod.add_bits<CuPyNumericBinaryOpCode>("BinaryOpCode",
                                        jlcxx::julia_type("CppEnum"));
  mod.set_const("ADD", CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_ADD);
  mod.set_const("ARCTAN2", CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_ARCTAN2);
  mod.set_const("BITWISE_AND",
                CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_BITWISE_AND);
  mod.set_const("BITWISE_OR",
                CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_BITWISE_OR);
  mod.set_const("BITWISE_XOR",
                CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_BITWISE_XOR);
  mod.set_const("COPYSIGN",
                CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_COPYSIGN);
  mod.set_const("DIVIDE", CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_DIVIDE);
  mod.set_const("EQUAL", CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_EQUAL);
  mod.set_const("FLOAT_POWER",
                CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_FLOAT_POWER);
  mod.set_const("FLOOR_DIVIDE",
                CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_FLOOR_DIVIDE);
  mod.set_const("FMOD", CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_FMOD);
  mod.set_const("GCD", CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_GCD);
  mod.set_const("GREATER", CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_GREATER);
  mod.set_const("GREATER_EQUAL",
                CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_GREATER_EQUAL);
  mod.set_const("HYPOT", CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_HYPOT);
  mod.set_const("ISCLOSE", CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_ISCLOSE);
  mod.set_const("LCM", CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_LCM);
  mod.set_const("LDEXP", CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_LDEXP);
  mod.set_const("LEFT_SHIFT",
                CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_LEFT_SHIFT);
  mod.set_const("LESS", CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_LESS);
  mod.set_const("LESS_EQUAL",
                CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_LESS_EQUAL);
  mod.set_const("LOGADDEXP",
                CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_LOGADDEXP);
  mod.set_const("LOGADDEXP2",
                CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_LOGADDEXP2);
  mod.set_const("LOGICAL_AND",
                CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_LOGICAL_AND);
  mod.set_const("LOGICAL_OR",
                CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_LOGICAL_OR);
  mod.set_const("LOGICAL_XOR",
                CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_LOGICAL_XOR);
  mod.set_const("MAXIMUM", CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_MAXIMUM);
  mod.set_const("MINIMUM", CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_MINIMUM);
  mod.set_const("MOD", CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_MOD);
  mod.set_const("MULTIPLY",
                CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_MULTIPLY);
  mod.set_const("NEXTAFTER",
                CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_NEXTAFTER);
  mod.set_const("NOT_EQUAL",
                CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_NOT_EQUAL);
  mod.set_const("POWER", CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_POWER);
  mod.set_const("RIGHT_SHIFT",
                CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_RIGHT_SHIFT);
  mod.set_const("SUBTRACT",
                CuPyNumericBinaryOpCode::CUPYNUMERIC_BINOP_SUBTRACT);
}