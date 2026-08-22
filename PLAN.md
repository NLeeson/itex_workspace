# Plan: Remapper must emit `ITEXLayerNorm` for Keras 2 FusedBatchNorm LN

## Revised finding (replaces FINDINGS Flaw 3 as the next slice)

FINDINGS Flaw 3 said “half-implemented SYCL LayerNorm” and prescribed a backward kernel plus Welford. Verbose runs show that is the wrong first cut.

**What the runtime actually does (locked by logs):**

- Keras 2 (`TF_USE_LEGACY_KERAS=1`) lowers `LayerNormalization` to `Reshape` + `FusedBatchNormV3` + `Mul`/`AddV2`.
- Remapper `layernorm` runs, sees `FusedBatchNormV3`, then fails: pattern wants `Fill` for unit scale/offset, the graph has **`Const` ones/zeros**.
- Even if Fill matched, `CheckInputOutputShape` rejects **rank 4** (`[64,128,128,128]`); the SYCL kernel already accepts ranks 2–4.
- Result: `gpu/fused_batch_norm_op.cc` → oneDNN `batch_normalization,ocl:simple` on Level Zero (`mb1ic64ih2097152`). That is the OpenCL-C path we do not want.
- `ITEXLayerNorm` / `LayerNormFwdSyclKernel` is registered on XPU and never selected.
- Keras 3 is a different graph (`Mean`/`SquaredDifference`/`Rsqrt`); remapper is off because XLA auto_jit. Out of this slice.
- Completing SYCL backward does not change these forward graphs.

**Revised statement:** Keras 2 last-axis LayerNorm is not migrated to the SYCL device variant because remapper `LayerNormFusion` does not match the current TF/Keras 2 FusedBatchNorm shape (Const vs Fill, rank 4, epsilon on the FBN attr). Fix the remapper so non-JIT XPU graphs emit `ITEXLayerNorm` and run `LayerNormFwdSyclKernel`.

## Why this is next

Single causal path from the last experiment. One fusion family, one dump we already have (`verbose_logs_ln/keras2_itex4/itex_optimizer*.pbtxt`). No adapter/threadpool work. No XLA lowering project.

## Non-goals (this PR)

- SYCL `LayerNormBwdSyclKernel` / Welford (old Flaw 3).
- Keras 3 moments fusion.
- `experimental_ops_override`.
- XLA custom-call for `ITEXLayerNorm`.
- Replacing all GPU `FusedBatchNormV3` / OpenCL-C BNorm (audit C). Real BN stays on FBN.

## Implementation (revised after 2026-08-19 review)

Compile-time remapper only. No kernel hot-path checks, no fallback, no walkers that reinterpret a rejected graph as last-axis LN.

1. **Eligible graph (must be proven, not assumed from the old dump)**  
   Build `LayerNormalization(axis=-1)` **on the rank-4 activation** with **unequal** dims (e.g. `[2, 8, 16, 32]`). Do not call the layer first on `[1, 128]`.  
   Eligible only if: plugin-runtime XPU; rank 2–4; FBN pre/post reshapes prove **last-axis** normalization; gamma and beta are already 1-D and equal the original last dim; FBN scale/offset are neutral splat `Fill` or `Const` (O(1) splat check, skip non-splat); epsilon copied from the FBN attr.

2. **Reject the recorded suffix-axis graph**  
   `test_single_op_ln.py` builds on `[1, 128]`, so Keras 2 stores `axis=1`. On `[64,128,128,128]` that is suffix-from-axis-1: FBN `NCHW` `[1,64,2097152,1]` (ones/zeros length **64**) and gamma/beta broadcast `[1,128,1,1]`. That is **not** last-axis LN. Broadcast `Reshape` on gamma/beta is a **rejection**, not a node to bypass. That workload may stay on `ocl:simple` this slice; record it as an unresolved XPU-invariant violation.

3. **Const and Fill**  
   Support folded `Const` ones/zeros **and** existing `Fill`. Same fusion, two input shapes.

4. **No gamma/beta walk-back**  
   `Update` feeds only already-1-D gamma/beta. Walking `Reshape` → `ReadVariable` would turn suffix-axis LN into last-axis LN; size checks can pass when all dims are 128.

5. **XLA**  
   Do **not** use `isXlaAutoJitEnabled()` as the guard: Keras 3 already sets `ITEX_REMAPPER=0` at NPD init; Keras 2 remapper stays on, and explicit `jit_compile=True` is a different scheduler path. Keep `ITEXLayerNorm` out of `XLA_GPU_JIT` at the graph-compile boundary (do not emit the custom op into a cluster that compiles as `XLA_GPU_JIT`). No feature-specific runtime branch.

6. **FINDINGS.md**  
   Rewrite Flaw 3 to remapper last-axis device-variant selection. Suffix-axis / Keras 3 moments / SYCL bwd / Welford stay later. Note: `ocl:simple` already uses `E[x²]−E[x]²`; Welford is not a gate.

## Files

- `itex/core/graph/remapper/layer_norm_pattern.cc` — last-axis eligibility only.
- `FINDINGS.md` — revised Flaw 3.
- `agentspace/test_single_op_ln.py` — build the layer at rank 4 with unequal last dim (verification workload, not a new Bazel matrix).

## Verify

Disk-only logs. Corrected workload (layer built on rank 4, unequal dims, `axis=-1`):

- Keras 2, `ITEX_TEST_JIT=0`: remapper emits `ITEXLayerNorm`; SYCL fwd kernel runs; no large `ocl:simple` BNorm; epsilon retained; one numeric compare vs the pre-fusion path or NumPy last-axis LN.
- Keras 2, `ITEX_TEST_JIT=1`: no `ITEXLayerNorm` on `XLA_GPU_JIT`.
- Old `[1,128]`-then-rank-4 script: still `FusedBatchNormV3` / `ocl:simple`; do not claim that case fixed.
- Keras 3: unchanged moments graph.

## Risk

Treating the existing dump as last-axis LN would ship a silent numerical bug. Eligibility is last-axis geometry only.

## (to review, reevaluate, double check and consideration), comments and concerns (2026-08-19):

**Status: accepted and folded into Implementation above.**

The direction is consistent with the domain invariant in [`CONTEXT.md`](./CONTEXT.md): an XPU graph must select a supported XPU device variant and must not select an OpenCL-C primitive. Therefore keeping the current `ocl:simple` fallback, or treating an OpenCL GPU ICD as the repository fix, is not an acceptable alternative. The CPU runtime ADRs remain out of scope: OpenMP versus Eigen THREADPOOL and loaded-library selection do not justify GPU device-variant behavior.

**Acceleration constraint:** add no per-execution validation, adapter, fallback branch, or kernel-side graph interpretation. The remapper is the compile step: it selects a kernel capability once from graph facts, then the selected kernel stays a direct fast path. Eligibility checks below are compile-time pattern guards, not runtime validation.

The plan does not yet prove that its proposed substitution preserves LayerNorm semantics:

1. **The recorded graph is not evidence for last-axis LayerNorm.** `test_single_op_ln.py` builds `LayerNormalization(axis=-1)` with a rank-2 `[1,128]` tensor, which resolves the stored axis to `1`, then reuses the layer on a rank-4 tensor. Legacy Keras consequently collapses and normalizes the suffix beginning at axis 1. The `[1,128,1,1]` gamma/beta reshapes in the dump are evidence of that broadcast contract, not removable noise.
2. **Do not walk back broadcast gamma/beta.** Feeding their 1-D producers to `LayerNormFwdSyclKernel` changes a suffix-axis normalization into last-axis normalization. In this dump all relevant dimensions equal 128, so the kernel's size checks can pass while values are wrong. A broadcast `Reshape` must be a rejection condition for this last-axis fusion, not a node to bypass.
3. **Prove the intended eligible graph separately.** Build the Keras layer directly on a rank-4 tensor with unequal dimensions and `axis=-1`. For an eligible match, gamma and beta must already be 1-D, must equal the original input's last dimension, and the FBN pre/post reshapes must prove the same last-axis normalization.
4. **Keep the compile contract minimal.** Supporting both `Fill` and folded `Const` is correct. Encode only facts needed to select `LayerNormFwdSyclKernel`: plugin-runtime XPU, rank 2–4, last-axis geometry, direct 1-D gamma/beta matching the last dimension, neutral FBN scale/offset, and copied epsilon. Prefer pattern structure and existing shape properties over new walkers. Accept scalar/splat `Fill` or `Const` in O(1); skip non-splat constants rather than scanning arbitrary tensors. Add no corresponding checks to the kernel hot path.
5. **Treat XLA as compile scheduling, not runtime validation.** `isXlaAutoJitEnabled()` describes global auto-JIT. It is redundant when initialization already disables the remapper and does not identify an explicit `jit_compile=True` cluster. Plugin-runtime remapping and `XLA_GPU_JIT` compilation are different scheduler paths: keep the custom op out of the latter at the graph-compile boundary, with no feature-specific runtime branch.
6. **Verify the compiled acceleration path; do not build a validation framework.** A new broad automated matrix or Bazel target is not a gate for this slice. Use the existing reproducible workload and captured graph artifacts, corrected so the layer is built at rank 4 with unequal dimensions. Success means the graph compiles, the remapper emits `ITEXLayerNorm`, the native SYCL kernel executes, the large OpenCL-C primitive disappears, epsilon is retained, and one independent output comparison confirms the substitution. If the existing focused test can express the folded-`Const` graph cheaply, extend it; otherwise keep verification as the recorded compile/run artifact.

Welford remains a valid later numerical-quality item, not a gate for this patch: the selected oneDNN `ocl:simple` fallback already uses the same `E[x²] - E[x]²` form. The present gate is semantic equivalence and correct device-variant selection.

This slice may legitimately cover only compile-time-proven last-axis Keras 2 graphs. If the recorded suffix-axis graph remains on `ocl:simple`, record it as an unresolved violation of the XPU invariant; do not count this slice as fixing that workload. Its eventual acceleration needs a kernel capability matching suffix-axis semantics, not more runtime validation around the last-axis kernel.
