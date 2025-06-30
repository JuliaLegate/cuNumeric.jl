
using cuNumeric: cuNumeric

function initialize(N)
    println("Initializing stencil grid...")
    grid = cuNumeric.zeros(Float64, N + 2, N + 2)
    grid[:, 1] = -273.15
    grid[:, end] = -273.15
    grid[end, :] = -273.15
    grid[1, :] = 40.0
    return grid
end

function run_stencil(N, I, warmup)
    grid = initialize(N)

    println("Running Jacobi stencil...")

    center = grid[2:(N + 1), 2:(N + 1)]
    north = grid[1:N, 2:(N + 1)]
    east = grid[2:(N + 1), 3:(N + 2)]
    west = grid[2:(N + 1), 1:N]
    south = grid[3:(N + 2), 2:(N + 1)]

    for i in 1:(I + warmup)
        average = center .+ north .+ east .+ west .+ south
        work = 0.2 .* average
        center = work
    end
end

run_stencil(1000, 100, 5)
