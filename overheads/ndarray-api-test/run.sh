export PATH=/home/david/FlameGraph:$PATH
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

FREQ=99
for script in ndarray.jl; do
  base=$(basename $script .jl)

  echo "Profiling $script ..."

  ENABLE_JITPROFILING=1 perf record -F $FREQ --call-graph dwarf julia --project="../../." $script

  perf script > ${base}.perf

  # Flamegraph
  stackcollapse-perf.pl ${base}.perf > ${base}.folded
  flamegraph.pl ${base}.folded > kernel_${base}.svg

  # Generate text callgraph with graphviz
  cat ${base}.perf  | c++filt | gprof2dot -f perf | dot -Tpdf -o ${base}_callgraph.pdf
done
