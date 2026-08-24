# ITEX XLA flags

ITEX reads XLA compilation options from the `TF_XLA_FLAGS` environment
variable. Flags are written as space-separated `--name=value` pairs:

`export TF_XLA_FLAGS="--tf_xla_use_device_api=true --tf_xla_auto_jit=2"`

```bash
TF_XLA_FLAGS="--tf_xla_auto_jit=1" python train.py
```

Multiple options can be combined:

```bash
TF_XLA_FLAGS="--tf_xla_auto_jit=1 --tf_xla_min_cluster_size=8" python train.py
```

## Automatic XLA compilation

### `tf_xla_auto_jit`

Controls automatic clustering of TensorFlow operations into XLA computations
on CPU and GPU devices. The default is `0`.

| Value | Behavior |
| --- | --- |
| `-1` | Disable automatic XLA compilation. |
| `0` | Use the `ConfigProto` setting. |
| `1` | Compile operations that are very likely to benefit from XLA. |
| `2` | Compile all eligible operations. |
| `fusible` | Compile only TensorFlow operations that XLA knows how to fuse. *(Experimental.)* |
| `single-gpu(<N>)` | Use optimization level `<N>` only for single-GPU graphs; use `0` otherwise. *(Experimental.)* |

For example:

```bash
# Conservative automatic compilation.
TF_XLA_FLAGS="--tf_xla_auto_jit=1" python train.py

# Compile all eligible operations.
TF_XLA_FLAGS="--tf_xla_auto_jit=2" python train.py

# Enable level 2 only for single-GPU graphs.
TF_XLA_FLAGS="--tf_xla_auto_jit=single-gpu(2)" python train.py
```

### Clustering limits and operation selection

| Flag | Default | Description |
| --- | ---: | --- |
| `tf_xla_min_cluster_size` | `4` | Minimum number of operations in an XLA cluster. Ignored for operations placed on an XLA device or explicitly marked for compilation. |
| `tf_xla_max_cluster_size` | Maximum `int32` | Maximum number of operations in an XLA cluster. |
| `tf_xla_ops_to_cluster` | *(empty)* | Limit clustering to named operations or shortcuts. Multiple values are comma-separated. |
| `tf_xla_cluster_exclude_ops` | *(empty)* | Exclude named operations from automatic clustering. Multiple values are comma-separated. |
| `tf_xla_clustering_fuel` | Maximum `int64` | Artificial limit on the number of operations eligible for clustering. |

Supported `tf_xla_ops_to_cluster` shortcuts are `PW`, `RED`, `MISC`, `PWRED`,
`REDUCEWINDOW`, `REDUCEWINDOWPW`, `BN`, and `FUSIBLE`. TensorFlow operation
names can also be specified, for example:

```bash
TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_ops_to_cluster=FUSIBLE,MatMul"
```

### Compilation behavior and diagnostics

| Flag | Default | Description |
| --- | --- | --- |
| `tf_xla_enable_lazy_compilation` | `true` | Enable lazy compilation. |
| `tf_xla_async_compilation` | `false` | Compile clusters in the background while the fallback path executes. |
| `tf_xla_always_defer_compilation` | `false` | Always defer compilation. |
| `tf_xla_compile_on_demand` | `false` | Compile operations one at a time just in time instead of using automatic clustering. |
| `tf_xla_cpu_global_jit` | `false` | Enable global CPU JIT compilation through `SessionOptions`. |
| `tf_xla_clustering_debug` | `false` | Dump graphs during XLA compilation. |
| `tf_xla_print_cluster_outputs` | `false` | Insert print nodes for values produced by XLA clusters. |
| `tf_xla_check_cluster_input_numerics` | `false` | Check all XLA cluster inputs with `CheckNumerics`. |
| `tf_xla_check_cluster_output_numerics` | `false` | Check all XLA cluster outputs with `CheckNumerics`. |
| `tf_xla_disable_constant_folding` | `false` | Disable constant folding before XLA compilation. |
| `tf_xla_deterministic_cluster_names` | `false` | Make auto-cluster function names deterministic across runs. |

### Persistent compilation cache

| Flag | Default | Description |
| --- | --- | --- |
| `tf_xla_persistent_cache_directory` | *(empty)* | Directory used to save and load JIT-compiled executables. |
| `tf_xla_persistent_cache_prefix` | `xla_compile_cache` | Prefix used for persistent cache entries. |
| `tf_xla_disable_strict_signature_checks` | `false` | Skip strict signature checks for entries loaded into the XLA compile cache. |

### MLIR bridge and graph dumping

| Flag | Default | Description |
| --- | --- | --- |
| `tf_mlir_enable_mlir_bridge` | *(automatic)* | Enable or disable the experimental MLIR-based TensorFlow compiler bridge. |
| `tf_mlir_bridge_safe_mode` | `false` | Restrict the MLIR bridge to graphs using currently supported features. |
| `tf_mlir_enable_merge_control_flow_pass` | `true` | Enable the MLIR `MergeControlFlow` pass. |
| `tf_mlir_enable_convert_control_to_data_outputs_pass` | `false` | Enable the MLIR control-to-data-outputs pass. |
| `tf_dump_graphs_in_tfg` | `false` | Dump transformed graphs in MLIR TFG dialect instead of `GraphDef`. |

### Floating-point jitter

These diagnostic options add a small amount of floating-point noise to named
tensors.

| Flag | Default | Description |
| --- | --- | --- |
| `tf_introduce_floating_point_jitter_to_tensors` | *(empty)* | Comma-separated tensor IDs in `<node name>:<output index>` format. |
| `tf_introduce_floating_point_jitter_amount` | `1e-5` | Amount added to each element of the selected tensors. |

### Deprecated option

`tf_xla_enable_xla_devices` defaults to `false` and generates `XLA_*` devices
that force compilation when used. This option is deprecated.

### Debugging-only safety overrides

The following flags default to `false` and are intended only for debugging.
Enabling either one is unsound:

- `tf_xla_disable_deadness_safety_checks_for_debugging`
- `tf_xla_disable_resource_variable_safety_checks_for_debugging`

Boolean flags accept `true` or `false`. Integer flags should be passed as
decimal numbers; list-valued flags use commas without spaces.
