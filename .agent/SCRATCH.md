# MatMul Brain: AVX2 / AVX_VNNI Performance Investigation

This is the durable evidence and knowledge notebook for the current CPU matmul regression hunt. Facts, commands, measurements, eliminated hypotheses, primitive anatomy, and tuning lessons will be added incrementally as they are verified.

## Mission

- Restore the historical fast CPU matmul path and determine exactly why current throughput is below 100 GFLOP/s.
- Then maximize measured throughput on the available AVX2/AVX_VNNI machine without corrupting numerical semantics or benchmark accounting.

## Benchmark contract

- Immutable comparison: `/home/user/workspace/agentspace/test_matmul_f32.py`.
- Raw square FP32 work is counted as `2 * N^3` FLOPs.
- The script synchronizes every iteration with `out.numpy()`, so its reported latency includes framework invocation and host materialization overhead in addition to GEMM execution.
- Default shape is `1024x1024 * 1024x1024`; this is about 2.147 GFLOP per iteration.
- Evidence hygiene: redirect all build output, verbose output, logging output, and other potentially heavy stdout/stderr to files before inspecting strictly bounded excerpts.

## Evidence ledger

- Historical user observations: peak about 1400 GFLOP/s; CPU and XPU operations around 400 GFLOP/s; current anomalous matmul below 100 GFLOP/s.
- Repository branch: `itex_dnnl_v313`.
- Recent build history includes a deliberate split between OpenMP and Eigen THREADPOOL oneDNN CPU libraries controlled by `ITEX_OMP_THREADPOOL`; this is an evidence-backed high-priority threading/loaded-library lead, not yet a conclusion.
- Confirmed regression: the August 20 CPU plugin's THREADPOOL MatMul path selected `single_thread_=1` through `ExecuteNThreadedGemm()` for the 1024-square workload. `CreateDnnlStream(..., 1)` consequently skipped oneDNN threadpool interop and constructed a plain stream, so the external Eigen pool was not attached.
- Confirmed repair mechanism already present in commit `b99a82f1`: CPU MatMul sets `single_thread_=-1`; the stream creates a synchronous adapter over the ITEX/TensorFlow-compatible Eigen pool.
- Controlled numbers on the same system and benchmark: old isolated wheel 111.33 GFLOP/s; current 16-thread adapter 343–373 GFLOP/s; current deliberately forced to one thread 108.36 GFLOP/s.
- Direct oneDNN baseline: `brg_matmul:avx2` on 1024-square FP32 averages 428 GFLOP/s and peaks at 504 GFLOP/s. This is the kernel ceiling observed before TensorFlow invocation and `.numpy()` materialization overhead.
- The standard CPU path is not using the GPU: verbose identifies `cpu,matmul,brg_matmul:avx2`; Level Zero appears only in XPU initialization and the separate CPU-JIT/XLA behavior.
- The current experimental commit is not shippable unchanged: the focused GPU MatMul target fails because runtime `std::is_same<Device, CPUDevice>::value` branches still reference members declared only under `INTEL_CPU_ONLY`. Restore compile-time CPU/GPU guards while retaining the CPU stream fix.
- Approved repair applied: five CPU-only branches in `matmul_op.h` now use compile-time `INTEL_CPU_ONLY` guards. Focused CPU and GPU MatMul targets both compile, while CPU keeps `single_thread_=-1` and therefore always attaches the intended external pool for non-OpenMP execution.
- Installed repaired AVX2 artifact hash: `e831e47aa32082f345da821f00eea9af9e9f4411e0b4ff185887dc8a174240a0`; immutable benchmark hash remains `6b091ce89727c317214cb45974e11f9a0d066a8b5b835a1ef12fb2de3bff0220`.

## Core distinction

Direct benchdnn answers whether the built oneDNN engine and its selected implementation can run quickly. The TensorFlow benchmark additionally tests ITEX dispatch, graph rewrites, tensor placement, threadpool interop, synchronization, and framework overhead. The first boundary where performance changes is more useful than a broad software inventory.

## Threadpool lesson

oneDNN built with `DNNL_RUNTIME_THREADPOOL` does not own a worker pool. Fast execution requires a stream created through threadpool interop with a live `threadpool_iface`. A plain stream may be valid yet effectively serial; the global verbose `nthr` info is not sufficient proof that a particular execution stream attached or used that many workers. The adapter creation/first-parallel-for evidence and execution latency are the decisive signals.

For 1024-square on this build, the first oneDNN parallel region reports eight work items. Measured effective-pool sweep: 1 thread 105.51, 4 threads 295.60, 8 threads 333.26, 12 threads 327.07, 16 threads 289.79 GFLOP/s. More configured threads cannot create more than eight jobs here and can increase scheduling/oversubscription cost. WSL reports eight cores with two logical CPUs per core; spreading an eight-worker pool across one sibling per core reaches 333.58–338.05 GFLOP/s, while packing it onto four SMT core pairs reaches 307.90 GFLOP/s.

The alternative oneDNN OpenMP runtime remains a live tuning path. A short controlled sweep found 16 threads with `KMP_AFFINITY=granularity=fine,scatter` and `KMP_BLOCKTIME=0` at 384.48 GFLOP/s (5.585 ms), ahead of the repaired Eigen THREADPOOL path. Replicate before adopting: short samples and WSL frequency/scheduler variance can reorder candidates.

Replication isolates the real settings: `KMP_BLOCKTIME=0` produced 392.38 and 392.41 GFLOP/s, while leaving blocktime unset produced 323.62 and 328.67. Sixteen OpenMP threads beat 13–15; setting TensorFlow's separate intra-op pool to one thread reached 397.36 GFLOP/s. `KMP_AFFINITY` is not a reliable lever under WSL because same-setting repeats span the compact/unset range.

## TensorFlow → ITEX → oneDNN route

The immutable raw workload begins as TensorFlow `MatMul` on `/CPU:0`. With CPU-default `ITEX_LAYOUT_OPT=0`, the unconditional NativeLayout pass rewrites it to `_ITEXMatMul`; runtime tracing shows `itex/core/kernels/cpu/matmul_op.cc` executing `MatMulOp<CPUDevice, float, float, float>`. That wrapper constructs a `dnnl::matmul`, and oneDNN selects `brg_matmul:avx2` with strict-FP32 `ab:ab:ab` descriptors and user scratchpad.

This route is identical under Keras 3.15.1 versus legacy `tf_keras` 2.16.0 and under `ITEX_REMAPPER=0` versus `1`. Remapper runs before layout rewriting and only changes matching fusion patterns; it does not own standalone MatMul redirection. No oneDNN Graph partition executed in the focused probe.

With `ITEX_LAYOUT_OPT=1`, OneDnnLayout instead rewrites to `_OneDnnMatMul`, registered as `OneDnnMatMulOp<CPUDevice, float>` in `itex/core/kernels/onednn/block/matmul_op.cc`. It selects the same valid `brg_matmul:avx2` primitive. This distinction explains why a route/configuration change can expose the historical native-wrapper heuristic even though primitive dispatch never changed: the regression lived above oneDNN, in stream construction by the native ITEX MatMul kernel.

## BRG versus JIT GEMM

oneDNN's next implementation after `brg_matmul:avx2` for the exact plain 1024-square FP32 descriptor is `gemm:jit:f32`. Benchdnn with 16 OpenMP threads and `KMP_BLOCKTIME=0` finds BRG at 526.63 GFLOP/s average / 606.00 best and JIT GEMM at 521.15 average / 589.91 best. Directly, BRG is marginally better.

Inside TensorFlow/ITEX, the ranking reverses. Paired 100-iteration immutable-script runs measured BRG at 382.74 and 378.07 versus JIT GEMM at 415.39 and 400.46 GFLOP/s. A slower WSL phase measured BRG at 339.82 and 378.63 versus GEMM at 377.12 and 374.06. Thus JIT GEMM has the higher observed peak and better resistance to the long slow tail, even though the isolated primitive averages are tied. Framework context can change the best implementation through scheduling, allocation alignment, and latency variance; never promote a benchdnn winner without an end-to-end A/B.

The native CPU wrapper now supports `ITEX_MATMUL_IMPL`: `auto` keeps oneDNN's first implementation, and an implementation substring such as `gemm` advances the primitive-descriptor iterator. With no variable set, promotion to JIT GEMM occurs only for plain rank-2 FP32→FP32 MatMul without bias or post-ops and only when oneDNN initially chose `brg_matmul:avx2`. This leaves fused, batched, reduced-precision, and other-ISA paths automatic. The escape hatch is `ITEX_MATMUL_IMPL=auto`.

## Primitive and dispatch anatomy

- Dense FP32 MatMul is `dst[M,N] = sum_k(src[M,K] * weights[K,N]) + optional_bias`; batched forms add leading broadcastable dimensions.
- Fully specified shapes at primitive creation enable better implementation/layout choice. Runtime dimensions are flexible but can reduce optimization opportunities.
- Plain layouts are generally preferred for MatMul; each input needs a contiguous useful axis, and destination should be plain with `N` contiguous. Use `any` only when a chosen/reordered representation can actually be cached and reused.
- On this CPU and shape, dispatch is `brg_matmul:avx2`. `ONEDNN_VERBOSE=dispatch` explains rejected candidates; `profile_exec` gives primitive time; `profile` adds creation/cache timing. Always read the emitted template because fields can vary by configuration.
- Verbose header `nthr` describes runtime-level maximum concurrency, not proof that a particular THREADPOOL stream has an attached pool. Correlate it with interop-adapter creation and measured execution.
- The given 99%-sparse COO workload dispatches to `ref:any`, not an optimized AVX2 sparse kernel: 2.175 ms average and 44.14 dense-equivalent GFLOP/s for `4x1,000,000 * 1,000,000x12`. This is a separate custom-kernel opportunity.

## Precision and fusion

- Default `fpmath_mode` is `strict`; optional `bf16`, `f16`, `tf32`, or `any` permits compute-time narrowing while FP32 storage remains FP32. Such modes change numerical behavior and cannot be claimed as strict-FP32 wins.
- AVX_VNNI accelerates dot-product-friendly reduced/integer data paths; strict FP32 throughput on this Alder Lake CPU is fundamentally an AVX2/FMA problem, not a VNNI instruction path.
- Primitive post-ops can fuse bias, eltwise (including GELU), sum, binary, and PReLU. Graph MatMul patterns allow optional BiasAdd, up to 20 unary/binary epilogue ops, optional Select, and optional float-to-float TypeCast.
- Fusion primarily saves intermediate writes/reads and framework launches; it does not increase the GEMM FLOP count. Report fused throughput with an explicit accounting convention.

## Current ceilings and shape behavior

- Direct dense square benchdnn min-time peaks observed: 256 -> 466, 512 -> 536, 1024 -> 532, 1536 -> 291, 2048 -> 450, 3072 -> 186 GFLOP/s. Average results were noisier, showing that thermal/frequency/WSL scheduling control is essential before comparing kernels.
- For 1024-square, direct oneDNN ~4.0–4.7 ms versus immutable TensorFlow benchmark ~5.75–6.25 ms on the fast THREADPOOL path. The remaining gap is framework dispatch plus forced `.numpy()` materialization/synchronization, not GEMM code alone.

## Library resolution is part of the kernel

A framework matmul can execute one CPU primitive while still carrying many
unrelated math/device runtimes in the same process. Always distinguish three
sets: the library that owns the primitive, libraries linked or dynamically
loaded into the process, and libraries that actually create workers or
background device state during the timed region.

On this installation, CPU `_ITEXMatMul` owns a packaged oneDNN v3.13.1
primitive. `/opt` oneMath 0.5.0 and `/opt` oneDNN are not loaded. ITEX's eager
XPU initialization nevertheless brings in oneMKL SYCL libraries, UR OpenCL and
Level Zero adapters, OpenCL/Level Zero drivers, TBB, and UMF. Thus an XPU-stack
change can perturb CPU scheduling without changing the primitive or device.

RUNPATH is lower priority than `LD_LIBRARY_PATH`. The oneDNN CPU DSO requests
`libiomp5.so` and carries a RUNPATH to its packaged Clang 22.1 runtime, but the
global oneAPI compiler path wins first. The bundled runtime improves the
immutable 1024-square benchmark by ~3.9% in paired tests. Likewise, UR 2026.1
ships UMF 1.1 but currently resolves custom `/opt/lib` UMF 1.2; selecting the
matched 1.1 improves paired throughput by ~10.7%. These are real secondary
losses, while the old sub-100 collapse remains the missing oneDNN threadpool
interop adapter.

Vectorization should be proven from generated code, not inferred from a build
label. `ONEDNN_JIT_DUMP=1` emits raw JIT binaries. The active
`gemm:jit:f32` route emits `jit_avx2_kernel_sgemm` kernels with 256-bit YMM
loads/broadcasts and dense `vfmadd231ps`; each inspected kernel contains 363
vector FMA instructions. oneMKL verbose independently reports its SGEMM target
as AVX-2. AVX_VNNI is present on the CPU but does not accelerate strict FP32
SGEMM; this path uses AVX2/FMA.

## 2026-08-23 pause checkpoint

The historical sub-100 collapse is closed: it was the native ITEX wrapper
constructing a plain oneDNN THREADPOOL stream after selecting
`single_thread_=1`, not a bad primitive, unsupported MatMul, Keras/remapper
change, or wrong device. The repaired CPU stream attaches the intended pool;
compile-time `INTEL_CPU_ONLY` guards keep the shared GPU instantiation valid.

The safe installed default is packaged oneDNN v3.13.1 with a narrowly scoped
plain-FP32 promotion from `brg_matmul:avx2` to `gemm:jit:f32`. The optional
`ITEX_MATMUL_BACKEND=mkl` route calls oneMKL 2026.1 LP64 SGEMM only for supported
plain rank-2 FP32 cases. It now skips oneDNN stream/setup work on repeated
validated nonzero shapes; cached zero-sized cases remain on the established
zero-fill path. Installed library SHA-256:
`23aac641f419ee32f5b111e53e76249047f96a6df24660067e66199db8e8a335`.
The previous installed binary is recoverable from
`.codex/evidence/libitex_cpu_internal_avx2.pre_cached_fastpath.17f978.so`.

Focused CPU, GPU, and installable-library builds pass. A final probe with both
backend and implementation variables unset selects oneDNN `gemm:jit:f32`; the
explicit backend emits three AVX2 MKL SGEMMs with no oneDNN MatMul. Both return
1024 for a 1024-square all-ones input. The MKL helper also passes NN, TN, NT,
and TT non-square cases
and alternating dynamic shapes against NumPy, including cached repeats.
Repeated `[0,4] x [4,3]` inputs return `[0,3]` zero outputs without an SGEMM.

Last performance evidence predates the newest cached shortcut: with bundled
OpenMP, matched UMF, and XPU hidden, oneDNN measured 397.93/413.79 and oneMKL
420.04/419.12 GFLOP/s, a roughly 3.4% MKL paired-average lead. Do not attribute
that lead to the cache shortcut until it receives a controlled immutable-script
A/B/A/B. Set `MKL_DYNAMIC=FALSE` in that comparison: a correctness smoke left
dynamic threading enabled and MKL chose eight workers despite a 16-thread cap.

Next shortest causal chain: reverse-replicate the UMF 1.1 versus custom 1.2
result and bundled versus system OpenMP result, then perform exactly one
controlled oneDNN/MKL A/B/A/B. Keep oneDNN default until those reversals are
reproducible. `KMP_BLOCKTIME=0` is established; `KMP_AFFINITY` is ruled out as
a meaningful WSL lever. Do not reopen Keras, remapper, primitive support,
Level Zero placement, or AVX2 vectorization without contradictory evidence.
