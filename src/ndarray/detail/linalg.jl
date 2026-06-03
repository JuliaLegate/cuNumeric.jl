function choose_nd_color_shape(shape::NTuple{N,Int}) where {N}
    color_shape = Base.ones(Int, N)
    if N > 2
        color_shape[1] = Legate.num_procs()
        done = false
        while !done && color_shape[1] % 2 == 0
            weight_per_dim = [shape[i] / color_shape[i] for i in 1:(N - 2)]
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

function prepare_manual_task_for_batched_matrices(full_shape::NTuple{N,Int}) where {N}
    initial_color_shape = choose_nd_color_shape(full_shape)
    tilesize = Tuple(
        (full_shape[i] + initial_color_shape[i] - 1) ÷ initial_color_shape[i] for i in 1:N
    )
    color_shape = Tuple((full_shape[i] + tilesize[i] - 1) ÷ tilesize[i] for i in 1:N)
    return tilesize, color_shape
end

function solve_batched(a::NDArray{T,N}, b::NDArray, x::NDArray) where {T,N}
    nrhs = size(b)[end]
    full_shape = size(a)
    tilesize_a, color_shape = prepare_manual_task_for_batched_matrices(full_shape)
    tilesize_b = (tilesize_a[1:(end - 1)]..., nrhs)

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

function svd_single(a::NDArray{T,N}, u::NDArray, s::NDArray, vh::NDArray) where {T,N}
    rt = Legate.get_runtime();
    lib = cuNumeric.get_lib();

    task = Legate.create_auto_task(rt, lib, cuNumeric.SVD);

    l_a = nda_to_logical_array(a)
    l_u = nda_to_logical_array(u)
    l_s = nda_to_logical_array(s)
    l_vh = nda_to_logical_array(vh)

    Legate.add_input(task, l_a)
    Legate.add_output(task, l_u)
    Legate.add_output(task, l_s)
    Legate.add_output(task, l_vh)

    Legate.add_broadcast(task, l_a)
    Legate.add_broadcast(task, l_u)
    Legate.add_broadcast(task, l_s)
    Legate.add_broadcast(task, l_vh)

    Legate.submit_auto_task(rt, task)
end

function _svd(a::NDArray{T,2}, full_matrices::Bool) where {T}
    m, n = size(a)
    k = min(m, n)
    u  = full_matrices ? zeros(T, m, m) : zeros(T, m, k)
    s  = zeros(T, k)
    vh = full_matrices ? zeros(T, n, n) : zeros(T, k, n)
    svd_single(a, u, s, vh)
    return u, s, vh
end
