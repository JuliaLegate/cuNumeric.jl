# MPI-launched (SPMD) bootstrap: `mpirun -np N julia ...`. Every rank is a Legate rank
# wired by Realm's MPI bootstrap plugin, so this path is mutually exclusive with the
# Distributed/addprocs p2p path (see ext/DistributedExt) -- addprocs refuses under MPI.

const _MPIInfo = NamedTuple{(:rank, :size, :kind),Tuple{Int,Int,Symbol}}
const _MPI_STATE = Ref{Union{Nothing,_MPIInfo}}(nothing)

# Rank/size/kind if an MPI launcher stamped this process, else nothing. Uses only
# MPI-specific vars -- SLURM_* is excluded because the addprocs/SlurmManager path also
# runs under SLURM without being an MPI job.
function _mpi_launch_env()
    if haskey(ENV, "OMPI_COMM_WORLD_SIZE")
        return (
            parse(Int, get(ENV, "OMPI_COMM_WORLD_RANK", "0")),
            parse(Int, ENV["OMPI_COMM_WORLD_SIZE"]),
            :ompi,
        )
    elseif haskey(ENV, "PMI_SIZE") # MPICH / Hydra, and srun+PMI
        return (parse(Int, get(ENV, "PMI_RANK", "0")), parse(Int, ENV["PMI_SIZE"]), :mpich)
    end
    return nothing
end

# Whether to bootstrap via MPI. `CUNUMERIC_BOOTSTRAP` forces the choice; the Distributed
# p2p path sets CUNUMERIC_DISTRIBUTED_WORKER before `using cuNumeric`, so it wins a tie.
function _detect_mpi_bootstrap()
    mode = get(ENV, "CUNUMERIC_BOOTSTRAP", "auto")
    (mode == "off" || mode == "p2p") && return nothing
    haskey(ENV, "CUNUMERIC_DISTRIBUTED_WORKER") && return nothing
    env = _mpi_launch_env()
    (mode == "mpi" && env === nothing) && return (0, 1, :mpich) # forced single rank
    return env
end

# Point Realm at the MPI bootstrap plugin. These are read when the runtime starts
# (ensure_runtime!), so setting them here in __init__ is early enough.
#
# NOTE: Legate's multi-node detection keys on OMPI_/MV2_/SLURM_ world-size vars, and it
# reads them when the Legate library loads -- before cuNumeric.__init__ runs -- so we
# cannot set them here. A MPICH (Hydra) launch only exposes PMI_*, so the launcher must
# surface OMPI_COMM_WORLD_* first (see examples/mpi/mpi_launcher.sh). OpenMPI and SLURM
# srun set them natively.
function _configure_mpi_bootstrap!(info)
    libdir = Legate.LEGATE_LIBDIR
    get!(ENV, "REALM_UCP_BOOTSTRAP_PLUGIN", joinpath(libdir, "realm_ucp_bootstrap_mpi.so"))
    get!(ENV, "REALM_UCP_BOOTSTRAP_MODE", "mpi")
    get!(ENV, "LEGATE_MPI_WRAPPER", joinpath(libdir, "liblegate_mpi_wrapper.so"))
    _MPI_STATE[] = (rank=info[1], size=info[2], kind=info[3])
    return nothing
end

"""Whether this process was launched under an MPI bootstrap (`mpirun -np N julia ...`)."""
under_mpi() = _MPI_STATE[] !== nothing

"""This process's MPI rank (0 when not under MPI)."""
mpi_rank() = under_mpi() ? _MPI_STATE[].rank : 0

"""Number of MPI ranks (1 when not under MPI)."""
mpi_size() = under_mpi() ? _MPI_STATE[].size : 1

"""Whether this is rank 0 -- the process that owns interactive/driver-side work."""
is_main_process() = mpi_rank() == 0

# The Distributed addprocs path cannot coexist with an MPI launch: all ranks are already
# Legate ranks, so there are no free processes for addprocs to configure.
function _assert_not_under_mpi()
    if under_mpi() || _detect_mpi_bootstrap() !== nothing
        error(
            "cuNumeric was launched under MPI (mpirun/mpiexec/srun); the Distributed " *
            "addprocs/init_workers path is disabled. Every rank is already a Legate rank " *
            "via the MPI bootstrap -- drop the MPI launcher to use addprocs instead.",
        )
    end
end
