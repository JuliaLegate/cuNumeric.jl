# Arithmetic demonstration ONLY: copies the current C++ launch formulas.
# Does not call the wrapper or prove an actual CUDA launch failure.
using Test
budget = 256
rows, cols = 65536, 1024
tx = min(budget, cols)
ty = min(div(budget, tx), rows)
grid_y = cld(rows, ty)
@show grid_y
@test grid_y <= 65535
