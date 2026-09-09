using cuNumeric, Test
a = cuNumeric.zeros(UInt8, 2^31 + 1)
expected = length(a)
actual = cuNumeric.nda_array_size(a)
@show expected actual
@test actual == expected
