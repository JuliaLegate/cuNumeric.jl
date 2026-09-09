# Single process, --cpus 2 --gpus 0 --omps 0 --sysmem 256 --numamem 0.
using cuNumeric, Test
cuNumeric.ensure_runtime!()
expected = 256 * 2^20
actual = cuNumeric.query_total_host_memory()
@show expected actual
@test actual == expected
