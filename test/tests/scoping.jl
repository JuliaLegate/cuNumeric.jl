using cuNumeric

function run_test(op, op_scope, FT, N)
    a = cuNumeric.rand(FT, (N, N))
    b = cuNumeric.rand(FT, (N, N))
    c_scoped = cuNumeric.zeros(FT, (N, N))

    c_base = op(a, b)

    cuNumeric.disable_gc!()
    op_scope(a, b, c_scoped)
    cuNumeric.init_gc!()

    return c_base, c_scoped
end

function run_slice_test(op, op_scoped, FT, N; f=0.04, k=0.06, dx=1.0)
    u = cuNumeric.rand(FT, (N, N))
    v = cuNumeric.rand(FT, (N, N))

    scoped = cuNumeric.zeros(FT, (N-2, N-2))
    args = (f=FT(f), k=FT(k), dx=FT(dx))

    base = op(u, v, args)

    cuNumeric.disable_gc!()
    op_scoped(u, v, scoped, args)
    cuNumeric.init_gc!()

    return base, scoped
end

binary_scope(op) = (a, b, out) -> @cunumeric out[:, :] = op(a, b)
slice_scope(op) = (u, v, out, args) -> @cunumeric out[:, :] = op(u, v, args)

const OPS = Dict(
    :add => (+),
    :negate_add => ((a, b) -> -a + b),
    :sub => (-),
    :mul => (*), :complex => ((a, b) -> (a + b) .* (a - b) .+ (-a .* b)),
)

const SLICE_OPS = Dict(
    :F_u => (
        (u, v, args) -> (
            -u[2:(end - 1), 2:(end - 1)] .*
            (v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)]) +
            args.f * (1 .- u[2:(end - 1), 2:(end - 1)])
        )
    ),
    :F_v => (
        (u, v, args) -> (
            u[2:(end - 1), 2:(end - 1)] .*
            (v[2:(end - 1), 2:(end - 1)] .* v[2:(end - 1), 2:(end - 1)]) -
            (args.f + args.k) * v[2:(end - 1), 2:(end - 1)]
        )
    ),
    :lap_u => (
        (u, _, args) -> (
            (
                u[3:end, 2:(end - 1)] .- 2*u[2:(end - 1), 2:(end - 1)] .+
                u[1:(end - 2), 2:(end - 1)]
            ) ./ args.dx^2 .+
            (
                u[2:(end - 1), 3:end] .- 2*u[2:(end - 1), 2:(end - 1)] .+
                u[2:(end - 1), 1:(end - 2)]
            ) ./ args.dx^2
        )
    ),
)

function run_all_ops(FT, N)
    results = Dict()

    # Regular binary/complex ops
    for (name, op) in OPS
        c_base, c_scoped = run_test(op, binary_scope(op), FT, N)
        results[name] = (c_base, c_scoped)
    end

    # Slice-heavy ops
    for (name, op) in SLICE_OPS
        c_base, c_scoped = run_slice_test(op, slice_scope(op), FT, N)
        results[name] = (c_base, c_scoped)
    end

    return results
end
