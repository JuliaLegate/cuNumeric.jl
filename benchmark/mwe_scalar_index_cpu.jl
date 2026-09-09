# Run read and write separately: the bad coordinate may abort the process.
using cuNumeric, Test
mode = only(ARGS)
@assert mode in ("read", "write")
n = 2^31 + 1
a = cuNumeric.zeros(UInt8, n) # About 2 GiB of host RAM.
a[n:n] .= UInt8(7)
@test only(Array(a[n:n])) == UInt8(7) # Slice control avoids scalar accessors.
println("Slice control passed; testing scalar $mode at $n"); flush(stdout)
if mode == "read"
    @test cuNumeric.@allowscalar(a[n]) == UInt8(7)
else
    cuNumeric.@allowscalar a[n] = UInt8(9)
    @test only(Array(a[n:n])) == UInt8(9)
end
