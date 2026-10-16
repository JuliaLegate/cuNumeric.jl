# Unit tests for MPI launcher detection + the addprocs mutual-exclusion guard.
# Pure ENV logic -- no runtime or real MPI launch needed. `withenv` isolates each case.
using Test

@testset "MPI launcher detection (_mpi_launch_env)" begin
    withenv("OMPI_COMM_WORLD_SIZE" => "4", "OMPI_COMM_WORLD_RANK" => "2",
        "PMI_SIZE" => nothing, "PMI_RANK" => nothing) do
        @test cuNumeric._mpi_launch_env() == (2, 4, :ompi)
    end
    withenv("OMPI_COMM_WORLD_SIZE" => nothing, "PMI_SIZE" => "3", "PMI_RANK" => "1") do
        @test cuNumeric._mpi_launch_env() == (1, 3, :mpich)   # MPICH exposes only PMI_*
    end
    withenv("OMPI_COMM_WORLD_SIZE" => "2", "OMPI_COMM_WORLD_RANK" => "0",
        "PMI_SIZE" => "9", "PMI_RANK" => "5") do
        @test cuNumeric._mpi_launch_env() == (0, 2, :ompi)    # OMPI wins when both present
    end
    withenv("OMPI_COMM_WORLD_SIZE" => nothing, "PMI_SIZE" => nothing) do
        @test cuNumeric._mpi_launch_env() === nothing         # not under a launcher
    end
end

@testset "bootstrap decision (_detect_mpi_bootstrap)" begin
    withenv("PMI_SIZE" => "2", "PMI_RANK" => "1",
        "CUNUMERIC_DISTRIBUTED_WORKER" => nothing, "CUNUMERIC_BOOTSTRAP" => nothing) do
        @test cuNumeric._detect_mpi_bootstrap() == (1, 2, :mpich)
    end
    withenv("PMI_SIZE" => "2", "CUNUMERIC_DISTRIBUTED_WORKER" => "1",
        "CUNUMERIC_BOOTSTRAP" => nothing) do
        @test cuNumeric._detect_mpi_bootstrap() === nothing   # p2p sentinel wins
    end
    for forced in ("off", "p2p")
        withenv("PMI_SIZE" => "2", "CUNUMERIC_BOOTSTRAP" => forced,
            "CUNUMERIC_DISTRIBUTED_WORKER" => nothing) do
            @test cuNumeric._detect_mpi_bootstrap() === nothing
        end
    end
    withenv("PMI_SIZE" => nothing, "OMPI_COMM_WORLD_SIZE" => nothing,
        "CUNUMERIC_BOOTSTRAP" => "mpi", "CUNUMERIC_DISTRIBUTED_WORKER" => nothing) do
        @test cuNumeric._detect_mpi_bootstrap() == (0, 1, :mpich)   # forced, single rank
    end
end

@testset "MPI helpers default (not under MPI)" begin
    @test cuNumeric.under_mpi() == false
    @test cuNumeric.mpi_rank() == 0
    @test cuNumeric.mpi_size() == 1
    @test cuNumeric.is_main_process() == true
end

@testset "addprocs guard (_assert_not_under_mpi)" begin
    withenv("OMPI_COMM_WORLD_SIZE" => nothing, "PMI_SIZE" => nothing,
        "CUNUMERIC_DISTRIBUTED_WORKER" => nothing, "CUNUMERIC_BOOTSTRAP" => nothing) do
        @test cuNumeric._assert_not_under_mpi() === nothing
    end
    withenv("OMPI_COMM_WORLD_SIZE" => "4", "OMPI_COMM_WORLD_RANK" => "0",
        "CUNUMERIC_DISTRIBUTED_WORKER" => nothing, "CUNUMERIC_BOOTSTRAP" => nothing) do
        @test_throws ErrorException cuNumeric._assert_not_under_mpi()
    end
    withenv("PMI_SIZE" => "4", "CUNUMERIC_DISTRIBUTED_WORKER" => "1",
        "CUNUMERIC_BOOTSTRAP" => nothing) do
        @test cuNumeric._assert_not_under_mpi() === nothing   # p2p sentinel -> not MPI
    end
end
