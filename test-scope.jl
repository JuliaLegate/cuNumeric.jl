using cuNumeric

cuNumeric.disable_gc!()

N = 1000
a = cuNumeric.rand(Float32, (N, N));
b = cuNumeric.rand(Float32, (N, N));
c = cuNumeric.zeros(Float32, (N, N));

iters = 1

for i in range(1, iters)
    @cunumeric begin
        d = (a[1:N, 1:N] .+ b[1:N, 1:N]) .* a[1:N, 1:N]
        c[:, :] = d .* a
    end
end
