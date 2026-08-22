# Level Zero and SYCL runtime flags

## Scope and test environment

These findings were collected on the Alder Lake-P Xe-LP GPU (`0x46a6`) with
the active oneAPI 2026.1 runtime. Although probes were compiled with the
2026.0 compiler, `LD_LIBRARY_PATH` selected the 2026.1 `libsycl` and Unified
Runtime (UR) adapters at execution time. Both active Level Zero adapters are
UR 0.12.0:

- Legacy: `libur_adapter_level_zero.so.0.12.0`
- V2: `libur_adapter_level_zero_v2.so.0.12.0`

The probe creates a Level Zero GPU queue, shared and device USM allocations,
two USM copies, and a kernel launch. Every configuration below completed with
the expected result unless stated otherwise. Results describe this machine and
runtime version; they are not general performance claims.

## Trace layers

Use both layers when diagnosing a flag:

```bash
SYCL_UR_TRACE=3 ZE_DEBUG=1 ./workload >ur-trace.log 2>ze-debug.log
```

`SYCL_UR_TRACE` writes to standard output and `ZE_DEBUG` writes Level Zero
driver diagnostics to standard error. Do not print either full log to the
terminal.

| `SYCL_UR_TRACE` | Verified behavior |
| --- | --- |
| `1` | Basic UR adapter and device discovery |
| `2` | UR API call tracing |
| `3` | Discovery plus UR API calls |
| `-1` | All supported trace levels |

UR tracing observes the public calls such as `urQueueCreate`,
`urEnqueueUSMMemcpy`, and `urEnqueueKernelLaunchWithArgsExp`. It does not
necessarily expose an adapter's final Level Zero implementation decision.
`ZE_DEBUG` is required to distinguish immediate versus regular command lists,
USM residency, and the concrete kernel-launch API.

## Adapter selection and behavior

`SYCL_UR_USE_LEVEL_ZERO_V2=0` selects the legacy adapter and
`SYCL_UR_USE_LEVEL_ZERO_V2=1` selects V2. `SYCL_UR_TRACE=1` identifies the
chosen platform as either `Level-Zero` or `Level-Zero V2`.

| Configuration | Verified Level Zero behavior |
| --- | --- |
| Legacy default | One immediate and one regular command list |
| Legacy + `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` | Two immediate lists and no regular list |
| V2 default | Four immediate, in-order command lists |
| V2 + `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=0` or `1` | Still four immediate, in-order lists; no effect |
| V2 + `UR_L0_V2_FORCE_BATCHED=1` | One immediate and two regular command lists |
| V2 + `UR_L0_V2_DISABLE_ZE_LAUNCH_KERNEL_WITH_ARGS=1` | Changes the final Level Zero call from `zeCommandListAppendLaunchKernelWithArguments` to `zeCommandListAppendLaunchKernel` |

The last two V2 controls are adapter-internal: their change is visible in
`ZE_DEBUG`, while the high-level UR trace remains the same.

## Legacy controls that changed runtime behavior

Use the `SYCL_PI_LEVEL_ZERO_*` names below when following the current LLVM
documentation. The installed legacy adapter also recognizes several older
`UR_L0_*` aliases, but the documented names are preferred.

| Variable | Observed legacy effect | Tuning implication |
| --- | --- | --- |
| `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` | Regular execution changes to immediate command lists | Compare against the default for latency-sensitive work; it removes batching and can reduce throughput. |
| `SYCL_PI_LEVEL_ZERO_USM_RESIDENT=0x000` | Removes all `zeContextMakeMemoryResident` calls from the probe | Consider only if default residency causes memory pressure; benchmark multi-device and allocation-heavy cases. |
| `SYCL_PI_LEVEL_ZERO_DISABLE_USM_ALLOCATOR=1` | Shared allocations drop 5→1 and device allocations 3→1 | Diagnostic/bisect setting. Keep allocator enabled for normal runs unless a proven allocator issue exists. |
| `SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=1` | Event pools increase 1→2 and device-only events are used | Candidate for asynchronous workloads; host waits can require proxy events, so benchmark end-to-end. |
| `UR_L0_MAX_NUMBER_OF_EVENTS_PER_EVENT_POOL=32` | A 432-event probe grows from 3 pools at the default to 14–15 pools; `=4` grows to 108–109 pools | Effective legacy capacity control. Do not lower it without measuring allocation overhead and peak event pressure. |
| `UR_L0_OOQ_INTEGRATED_SIGNAL_EVENT=0/1` | `=0` issued about 384 explicit signal-event commands, 194 barriers, and 145 submissions; `=1` used no explicit signal-event commands, about 49 barriers, and 64 submissions | Effective legacy submission-topology control. The `=1` result appears to use driver-integrated signaling, not no signaling. Benchmark correctness and asynchronous throughput before selecting either mode. |

## V2 controls and limitations

| Variable | Result on this system | Guidance |
| --- | --- | --- |
| `UR_L0_V2_FORCE_BATCHED=1` | Effective; switches much of execution to regular command lists | The only performance-oriented V2 toggle demonstrated here. Compare throughput and latency against V2 default. |
| `UR_L0_V2_DISABLE_ZE_LAUNCH_KERNEL_WITH_ARGS=1` | Effective; selects the older launch path | Use only for compatibility investigation or bisection, not routine tuning. |
| `UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1` | No visible change | This Xe-LP reports no main or link blitter/copy engine, so copy offload cannot be exercised. |
| `SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=1` | No visible change | V2 uses immediate command lists; this legacy event feature is therefore ineffective here. |
| `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` | No visible change | V2 currently remains immediate-list based on this workload. |

The current LLVM documentation describes V2 as optimized for queue modes, but
also notes that it currently supports immediate command lists. On this older
Xe-LP platform, keep legacy as the baseline and treat V2 as an explicit,
benchmark-gated experiment.

## Unsupported or workload-limited controls

- `SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=1` and
  `UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1` cannot alter this GPU's behavior:
  the Level Zero driver reports that main and link blitter/copy engines are
  unavailable.
- `SYCL_UR_L0_RESTRICT_USM_RESIDENCY_TO_P2P` is documented as V2-only but is
  absent from both installed UR 0.12 adapter binaries. Treat it as unsupported
  in this runtime.
- `UR_L0_QUEUE_SYNCHRONIZE_NON_BLOCKING` is recognized by both adapters, but
  has not been classified here: call counts vary and do not establish whether
  a queue synchronization blocks. Test it with a wait/timeout latency probe.
- `UR_L0_SERIALIZE` is recognized by both adapters, but has not been
  classified here: it changes locking and blocking around UR calls, which
  requires a contended multi-thread queue/allocation test and timing or lock
  profiling.
- `UR_L0_USE_MULTIPLE_COMMANDLIST_BARRIERS` is legacy-only. A probe with 48
  explicit SYCL barriers showed no reproducible topology change, so it is
  workload-conditional rather than proven inert.
- `SYCL_PI_LEVEL_ZERO_USE_COMPUTE_ENGINE`, copy batching, native 2D memcpy,
  event-recycling thresholds, and similar knobs require a targeted workload
  before being classified as beneficial or ineffective. Do not enable them
  based on static string presence alone.
- `SYCL_UR_L0_DRIVER_SKIPLIST=1.15.0` is effective for both adapters: it
  removes the matching Level Zero GPU. It is a compatibility exclusion control,
  not a tuning control.

## NEO debug keys

```bash
NEOReadDebugKeys=1 PrintDebugMessages=1 ./workload
```

`NEOReadDebugKeys=1` is required for the release driver to honor NEO debug
keys. With it, `PrintDebugMessages=1` emitted driver diagnostics including
scratch compute-unit and hardware information. `PrintDebugMessages=1` alone
did not. `EnableSharedSystemUsmSupport=1` showed no change in this workload;
do not treat it as a performance knob without a capability-specific test.

NEO and SYCL debug variables are diagnostic interfaces whose behavior can
change between releases. Never include them in a production default
environment.

## Recommended tuning workflow

1. Start with no Level Zero or NEO override and record correctness and
   end-to-end throughput/latency.
2. Select one adapter explicitly for each comparison. Legacy is the baseline
   on this Xe-LP system; compare V2 only with
   `SYCL_UR_USE_LEVEL_ZERO_V2=1`.
3. For legacy, test one of immediate lists, device-scope events, USM
   residency, event-pool capacity, or OOO integrated signaling at a time.
   Keep the USM allocator enabled by default.
4. For V2, compare its default against `UR_L0_V2_FORCE_BATCHED=1`. Use the
   launch-with-args switch only to investigate a correctness or driver issue.
5. Use a contended multi-thread test before evaluating `UR_L0_SERIALIZE`, and
   a wait/timeout latency test before evaluating
   `UR_L0_QUEUE_SYNCHRONIZE_NON_BLOCKING`.
6. Collect `SYCL_UR_TRACE=3` and `ZE_DEBUG=1` only during diagnosis, to files,
   then retest without tracing before judging performance.
7. Do not tune copy engines on this Xe-LP GPU: hardware capability makes those
   flags inert here.

## References

- [LLVM SYCL environment variables](https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md)
- [Unified Runtime Level Zero adapter variables](https://oneapi-src.github.io/unified-runtime/core/LEVEL_ZERO.html)
- [Intel Compute Runtime FAQ](https://github.com/intel/compute-runtime/blob/master/FAQ.md)

- [MORE FLAGS] (https://intel.github.io/llvm/EnvironmentVariables.html)
