@testset "NAS EP contract" begin
    @test NAS_EP_NPB_GPU_COMMIT == "3f12d84920ee315ab00ef283717c1e74b68f4d00"
    @test Set(keys(NAS_EP_CLASSES)) == Set(["S", "W", "A", "B", "C", "D", "E"])
    b = NASEmbarrassinglyParallel{Float64}(; N=33_554_432, M=1, class="S")
    p = validate_nas_ep(b)
    @test p == nas_ep_parameters("s")
    @test p.m == 24
    @test nas_ep_batches(p) == 65_536
    @test nas_ep_pairs_per_batch() == 256
    @test total_flops(b) == 33_554_432.0
    @test total_space(b) == 15_204_352
    @test nas_ep_verified("S", p.sx, p.sy)
    @test !nas_ep_verified("S", 1.01p.sx, p.sy)
    first_batch = nas_ep_batch(0)
    @test first_batch.q0 == 97
    @test first_batch.q1 == 86
    @test first_batch.sx ≈ -9.836968663257105
    class_s = nas_ep_combine(
        nas_ep_batch(i) for i in 0:(nas_ep_batches(p) - 1)
    )
    @test class_s.q == [6_140_517, 5_865_300, 1_100_361, 68_546, 1_648, 17, 0, 0, 0, 0]
    @test nas_ep_verified("S", class_s.sx, class_s.sy)
    @test_throws ErrorException validate_nas_ep(
        NASEmbarrassinglyParallel{Float32}(; N=33_554_432, M=1, class="S")
    )
    @test all(
        supports_benchmark(execution_model(model), "nas_ep") for
        model in (:cunumeric, :cupynumeric, :cudajl, :jacc, :dagger)
    )
    @test supports_run(execution_model(:jacc), "nas_ep", 2)
    @test supports_run(execution_model(:dagger), "nas_ep", 2)

    config = joinpath(@__DIR__, "..", "benchmarks_nas_ep.toml")
    settings, specs = parse_config(config)
    runs = plan_runs(
        specs, settings, TOML.parsefile(config), parse_plot_groups(config), 10^12
    )
    @test Set(r.model for r in runs) ==
        Set([:cunumeric, :cupynumeric, :cudajl, :jacc, :dagger])
    @test all(r.N == 33_554_432 && r.M == 1 && r.spec.n_iter == 1 for r in runs)
end

@testset "NAS FT contract" begin
    @test NAS_FT_NPB_GPU_COMMIT == "3f12d84920ee315ab00ef283717c1e74b68f4d00"
    @test Set(keys(NAS_FT_CLASSES)) == Set(["S", "W", "A", "B", "C", "D", "E"])
    b = NASFourierTransform{Float64}(; N=64, M=64, class="S")
    p = validate_nas_ft(b)
    @test p == nas_ft_parameters("s")
    @test p.nz == 64 && p.niter == 6
    @test total_flops(b) ≈ 1.7716695575300533e8
    @test total_space(b) == 16_777_216
    @test length(NAS_FT_CHECKSUMS["S"]) == p.niter
    @test length(nas_ft_checksum_indices(p)) == NAS_FT_CHECKSUM_SAMPLES
    @test sum(nas_ft_checksum_mask(p)) == NAS_FT_CHECKSUM_SAMPLES
    initial = Array{ComplexF64}(undef, 2, 2, 1)
    nas_ft_initial_conditions!(initial)
    @test initial[1] ≈ 0.7945219111887383 + 0.8690652738745399im
    @test_throws ErrorException validate_nas_ft(
        NASFourierTransform{Float32}(; N=64, M=64, class="S")
    )
    @test all(
        supports_benchmark(execution_model(model), "nas_ft") for
        model in (:cunumeric, :cupynumeric, :cudajl, :jacc, :dagger)
    )
    @test !supports_run(execution_model(:jacc), "nas_ft", 2)

    config = joinpath(@__DIR__, "..", "benchmarks_nas_ft.toml")
    settings, specs = parse_config(config)
    runs = plan_runs(
        specs, settings, TOML.parsefile(config), parse_plot_groups(config), 10^12
    )
    @test Set(r.model for r in runs) ==
        Set([:cunumeric, :cupynumeric, :cudajl, :jacc, :dagger])
    @test all(r.N == 64 && r.M == 64 && r.spec.n_iter == 1 for r in runs)
end
