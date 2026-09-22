import sys, time, statistics
import numpy as np
import cupynumeric as cn
from legate.core import get_legate_runtime
n = int(sys.argv[1])
rt = get_legate_runtime()
def sync():
    rt.issue_execution_fence(block=True)
print('cupynumeric', cn.__version__, 'n', n, flush=True)
A = cn.ones((n,n), dtype=cn.float32)
print("Input constructed", type(A._thunk).__name__, flush=True)
x = cn.ones(n, dtype=cn.float32)
y = cn.empty(n, dtype=cn.float32)
sync()
print("Initialized", flush=True)
for _ in range(3):
    cn.dot(A, x, out=y)
    sync()
print("Warmed", flush=True)
samples=[]
for _ in range(5):
    start=time.perf_counter()
    cn.dot(A,x,out=y)
    sync()
    samples.append(1000*(time.perf_counter()-start))
assert np.all(np.asarray(y)==np.float32(n))
print('RESULT',n,'median_ms',statistics.median(samples),'samples',samples,flush=True)
with open('/proc/self/maps') as f:
    print('CUBLAS_LIBRARIES', sorted({line.split()[-1] for line in f if 'libcublas' in line}),flush=True)
