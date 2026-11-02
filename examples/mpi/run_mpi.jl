# SPMD cuNumeric under MPI. Launch with (see README.md):
#   SLURM:  sbatch run.sh
#   local:  mpirun -np 4 ./mpi_launcher.sh env LEGATE_CONFIG="--cpus 2" julia --project=. run_mpi.jl
using cuNumeric
cuNumeric.Experimental(true)

rank = cuNumeric.mpi_rank()
size = cuNumeric.mpi_size()
cuNumeric.is_main_process() && println("cuNumeric MPI run across $size ranks")

a = cuNumeric.rand(1_000)
total = cuNumeric.sum(a)  # global reduction, identical on every rank
println("rank=$rank/$size  sum=$total")
