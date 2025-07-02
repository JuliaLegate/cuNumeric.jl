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

#include "legate.h"
#include "cupynumeric.h"
#include <vector>
#include <iostream>


// using cupynumeric::slice;

void matmul_fp32(size_t N){

    std::vector<uint64_t> dims = {N,N};
    auto A = cupynumeric::random(dims).as_type(legate::float32());
    auto B = cupynumeric::random(dims).as_type(legate::float32());
    auto C = cupynumeric::zeros(dims, std::optional<legate::Type>(legate::float32()));

    C.dot(A,B);

    // std::cout << C[{slice(0,0), slice(0,0)}] << std::endl;
}


int main(int argc, char** argv){
    auto result = legate::start(argc, argv);
    assert(result == 0);

    cupynumeric::initialize(argc, argv);

    const size_t N = 10000;
    matmul_fp32(N);

    return legate::finish();
}
