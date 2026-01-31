#= Copyright 2026 Northwestern University, 
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
=#

function Experimental(setting::Bool)
    task_local_storage(:Experimental, setting)
end

function assert_experimental()
    if get(task_local_storage(), :Experimental, false) !== true
        throw(
            ArgumentError(
                "Experimental features are disabled." *
                " Use `cuNumeric.Experimental(true)` to enable them.",
            ),
        )
    end
end

function _test_task(args::Vector{Legate.TaskArgument})
    a, b, scalar = args
    @inbounds @simd for i in eachindex(a)
        b[i] = a[i] * scalar
    end
end

function test_task_interface(in_arr::NDArray{T}, scalar::Float32) where {T}
    assert_experimental()

    out_arr = cuNumeric.zeros(T, Base.size(in_arr))

    rt = Legate.get_runtime()
    lib = get_lib()

    my_task = Legate.wrap_task(_test_task)

    task = Legate.create_julia_task(rt, lib, my_task)

    input_vars = Vector{Legate.Variable}()
    output_vars = Vector{Legate.Variable}()

    push!(input_vars, Legate.add_input(task, get_store(in_arr)))
    push!(output_vars, Legate.add_output(task, get_store(out_arr)))

    Legate.default_alignment(task, input_vars, output_vars)
    Legate.add_scalar(task, Legate.Scalar(scalar))

    Legate.submit_task(rt, task)

    # In single-threaded mode, we must wait to avoid deadlock
    # (Main thread blocks on data access, preventing async_worker from running)
    if Threads.nthreads() == 1
        Legate.wait_ufi()
    end

    return out_arr
end
