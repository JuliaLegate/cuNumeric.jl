using Test, Pkg, CNPreferences

@testset "linear algebra preferences" begin
    mktempdir() do dir
        # Preference writes must not alter the developer's active project.
        write(joinpath(dir, "Project.toml"),
            "[deps]\nCNPreferences = \"$(Base.PkgId(CNPreferences).uuid)\"\n")
        previous = Base.active_project()
        try
            Pkg.activate(dir; io=devnull)
            settings = (
                MIN_SOLVE_MATRIX_SIZE=32, MIN_SOLVE_TILE_SIZE=4,
                MIN_CHOLESKY_MATRIX_SIZE=32, MIN_CHOLESKY_TILE_SIZE=4,
                MIN_QR_MATRIX_SIZE=1, QR_TILE_SIZE=4, MAX_CHOLESKY_TILES_PER_PROC=2,
            )
            CNPreferences.set_linalg!(; settings...)
            for (key, value) in pairs(settings)
                @test cuNumeric.load_preference(CNPreferences, string(key)) == value
            end
            path = joinpath(dir, "LocalPreferences.toml")
            saved = read(path, String)
            for bad in (0, -1, true, 1.5, "4")
                @test_throws ArgumentError CNPreferences.set_linalg!(;
                    MIN_SOLVE_MATRIX_SIZE=100, QR_TILE_SIZE=bad
                )
                @test read(path, String) == saved
            end
            @test_throws ArgumentError CNPreferences.set_linalg!(; QR_TIEL_SIZE=4)
            @test read(path, String) == saved
        finally
            Pkg.activate(dirname(previous); io=devnull)
        end
    end
end
