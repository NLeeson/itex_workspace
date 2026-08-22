# ITEX execution

Intel Extension for TensorFlow runs TensorFlow graphs with oneDNN primitives and ITEX kernels. **CPU oneDNN runtime** (OpenMP vs Eigen THREADPOOL) and **GPU oneDNN runtime** (SYCL) are different products. They may share an **orchestration** shape; they do not share workers.

## Language

### Runtimes

**CPU oneDNN runtime**:
Which oneDNN CPU library this process actually opened: **OpenMP** or **Eigen THREADPOOL**. Graph rewrite and stream creation use that frozen load, not a later reread of the env var.
_Avoid_: threadpool (bare), OMP threadpool, unification, enable_omp

**OpenMP**:
The CPU oneDNN runtime that owns an OpenMP worker team (`libiomp5`). It does not use the TensorFlow intra-op pool.
_Avoid_: OMP threadpool

**Eigen THREADPOOL**:
The CPU oneDNN runtime that has no workers of its own. It only computes by submitting work through a **oneDNN THREADPOOL adapter**.
_Avoid_: unified threadpool, TF threadpool (as a name for this runtime)

### Pools and adapters

**TensorFlow intra-op pool**:
The Eigen worker team TensorFlow owns for intra-op parallelism and threading knobs.
_Avoid_: the threadpool, Eigen device (unqualified)

**Legacy ITEX pool**:
ITEX’s process-wide Eigen `ThreadPool` singleton. Intel ITEX still uses it for `eigen_cpu_device()`. Together with the TensorFlow intra-op pool (and, on OpenMP, the OpenMP team) this is the **two-pool problem**.
_Avoid_: unified threadpool, the ITEX threadpool (unqualified)

**OpenMP team**:
The worker team owned by OpenMP when the CPU oneDNN runtime is OpenMP.

**oneDNN THREADPOOL adapter**:
The `threadpool_iface` implementation that lets Eigen THREADPOOL oneDNN run on an existing framework pool. In ITEX that pool is the TensorFlow intra-op pool.
_Avoid_: unified threadpool, TF-oneDNN threadpool

**oneDNN Graph**:
A graph partitioner that turns TensorFlow subgraphs into oneDNN partitions. It does not choose who owns workers.

**GPU oneDNN runtime**:
oneDNN built for SYCL, bound to ITEX’s device queue via `sycl_interop`. SYCL is the preferred device backend this project runs.
_Avoid_: SYCL as a name for the CPU oneDNN THREADPOOL adapter, SYCL parallel_for as CPU intra-op

**Orchestration**:
Submit work, then wait at the stream or primitive boundary where the host needs results. SYCL queues do this; Eigen THREADPOOL oneDNN should do the same through the adapter (`parallel_for` submits, `wait()` joins). OpenMP oneDNN is fork-join inside `execute`, not this shape.
_Avoid_: SYCL parallel_for (as a synonym for the CPU adapter), unified threadpool

### Call direction

**Submit onto**:
The only legal integration between Eigen THREADPOOL oneDNN and TensorFlow: oneDNN calls the adapter; the adapter posts jobs onto the TensorFlow intra-op pool.
_Avoid_: TF schedules through oneDNN, unified threadpool, compatible as peers

### Graph rewrite

**Native-layout rewrite**:
A Grappler rewrite that replaces a TensorFlow op with an ITEX native kernel (`_ITEX*`). It is not a compute backend and does not own workers.

**OpenMP native-layout extras**:
The native-layout rewrites that belong only when the CPU oneDNN runtime is OpenMP (`AddN`, `Elu`, `EluGrad`, `LeakyRelu`, `LeakyReluGrad`, `Relu6Grad`, `ReluGrad`, `ResizeBilinear`, `ResizeBilinearGrad`, `Slice`).
_Avoid_: dropped ops, Eigen native format (as a name for this set)

**Shared native-layout table**:
The native-layout rewrites that apply on both CPU oneDNN runtimes.

### Devices and kernel variants

**XPU**:
The Next Pluggable Device name for Intel GPU in TensorFlow (`/device:XPU:0`). It is not the CPU device.

**Device variant**:
The kernel implementation registered for an op on one TensorFlow device: CPU oneDNN, GPU native SYCL, or GPU oneDNN. A graph on CPU must not pick an XPU variant; a graph on XPU must not pick a CPU variant or an **OpenCL-C primitive**.
_Avoid_: wrong device (unqualified), the GPU kernel (when CPU and XPU differ)

**OpenCL-C primitive**:
A oneDNN GPU primitive that compiles OpenCL-C and needs an OpenCL ICD. ITEX XPU is Level Zero only; that path is unsupported.
_Avoid_: OpenCL backend (as if ITEX shipped it)

**XLA_GPU_JIT**:
The TensorFlow compilation device for `@tf.function(jit_compile=True)` on XPU (NPD sets `compilation_device_name`). It is not `DEVICE_GPU`. An **ITEX custom op** has no kernel there unless an XLA lowering exists.
_Avoid_: XLA as a synonym for the SYCL OpKernel, JIT (unqualified)

**ITEX custom op**:
A fused ITEX node such as `ITEXLayerNorm`. Remapper and ops-override emit it for the plugin runtime. XLA cannot compile it unless a lowering or rewrite-suppression pass keeps the cluster in HLO.
