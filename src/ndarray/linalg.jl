function choose_nd_color_shape(shape::NTuple{N,Int}) where N
    color_shape = Base.ones(Int, N)
    if N > 2
            color_shape[1] = Legate.num_procs()
        done = false
        while !done && color_shape[1] % 2 == 0
            weight_per_dim = [shape[i] / color_shape[i] for i in 1:N-2]
            max_weight, idx = findmax(weight_per_dim)
            if weight_per_dim[idx] > 2 * weight_per_dim[1]
                color_shape[1] ÷= 2
                color_shape[idx] *= 2
            else
                done = true
            end
        end
    end
    return Tuple(color_shape)
end

function prepare_manual_task_for_batched_matrices(full_shape::NTuple{N,Int}) where N
    initial_color_shape = choose_nd_color_shape(full_shape)
    tilesize = Tuple((full_shape[i] + initial_color_shape[i] - 1) ÷ initial_color_shape[i] for i in 1:N)
    color_shape = Tuple((full_shape[i] + tilesize[i] - 1) ÷ tilesize[i] for i in 1:N)
    return tilesize, color_shape
end

function solve_batched(a::NDArray{T,N}, b::NDArray, x::NDArray) where {T,N}
    nrhs = size(b)[end]
    full_shape = size(a)
    tilesize_a, color_shape = prepare_manual_task_for_batched_matrices(full_shape)
    tilesize_b = (tilesize_a[1:end-1]..., nrhs)

    store_a = nda_to_logical_store(a)
    store_b = nda_to_logical_store(b)
    store_x = nda_to_logical_store(x)

    tiled_a = Legate.partition_by_tiling(store_a, collect(tilesize_a))
    tiled_b = Legate.partition_by_tiling(store_b, collect(tilesize_b))
    tiled_x = Legate.partition_by_tiling(store_x, collect(tilesize_b))

    rt = Legate.get_runtime()
    domain = Legate.domain_from_shape(Legate.Shape(Legate.to_cxx_vector(color_shape)))
    lib = cuNumeric.get_lib()
    task = Legate.create_manual_task(rt, lib, cuNumeric.SOLVE, domain)

    Legate.add_input(task, tiled_a)
    Legate.add_input(task, tiled_b)
    Legate.add_output(task, tiled_x)

    Legate.submit_manual_task(rt, task)
end

function solve(a::NDArray{T,N}, b::NDArray{S,M}) where {T,N,S,M}
    if N < 2
        throw(ArgumentError("$(N)-dimensional array given. Array must be at least two-dimensional"))
    end
    if M < 1
        throw(ArgumentError("$(M)-dimensional array given. Array must be at least one-dimensional"))
    end
    if T == Float16 || S == Float16
        throw(ArgumentError("array type float16 is unsupported in linalg"))
    end
    if size(a)[end-1] != size(a)[end]
        throw(ArgumentError("Last 2 dimensions of the array must be square"))
    end
    if N == 2 && size(a)[2] != size(b)[1]
        if M == 1
            throw(ArgumentError(
                "Input operand 1 has a mismatch in its dimension 0, " *
                "with signature (m,m),(m)->(m) (size $(size(b)[1]) " *
                "is different from $(size(a)[2]))"
            ))
        else
            throw(ArgumentError(
                "Input operand 1 has a mismatch in its dimension 0, " *
                "with signature (m,m),(m,n)->(m,n) (size $(size(b)[1]) " *
                "is different from $(size(a)[2]))"
            ))
        end
    end
    if N > 2
        if N != M
            throw(ArgumentError(
                "Batched matrices require signature (...,m,m),(...,m,n)->(...,m,n)"
            ))
        end
        if size(a)[end] != size(b)[end-1]
            throw(ArgumentError(
                "Input operand 1 has a mismatch in its dimension " *
                "$(M-2), with signature (...,m,m),(...,m,n)->(...,m,n)" *
                " (size $(size(b)[end-1]) is different from $(size(a)[end]))"
            ))
        end
    end
    if prod(size(a)) == 0 || prod(size(b)) == 0
        return zeros(T, size(b)...)
    end
    x = zeros(T, size(b)...)
    solve_batched(a, b, x)
    return x
end