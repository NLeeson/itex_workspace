ITEX was built with oneDNN 3.13 and CPU OMP and GPU SYCL


I want to know precicely why tests like this
`python test/python/lamb_optimizer_test.py`

fail or pass depending on the mode we set ITEX.
Specifically the example give would pass when invoked ith `ITEX_REMAPPER=0` 
Why?

Also take a look into the mixed precision which tries to access the filesystem and fails countless times while a few times it works and replaces some operations.
For the Graph it seems not to have an effect at all, oneDNN prints itself with different backend settings etc. investigate the behavior this follows.
The difference in disabling XLA and/or switching keras mode are having the highest effect on visible verbosity prints describing a different execution.


# Flags that have effect on the test result and behavior

# Verbosity (produces alot of output scope tests well to not overload your context)
`ITEX_VERBOSE=1`
`ONEDNN_VERBOSE=all`


# Toggle for the following is VAR=0|1

# disable itex XLA (does NOT disable the keras jit that is default in K3)
`ITEX_DISABLE_XLA=1`

# use Keras 2 (no default jit, keras 2 and 3 are using different execution models)
`TF_USE_LEGACY_KERAS=1`
* more on keras version and xla 
`./docs/guide/keras3_support.md`

`ITEX_REMAPPER=1`

`ITEX_LAYOUT_OPT=1`

# Switch between NPD or Pluggable Device (XLA depends on NPD) 
`ITEX_ENABLE_NEXTPLUGGABLE_DEVICE=1`

# Ops Override overloads optimized layer implementations on the namespace of standard layers replacing them
`ITEX_OPS_OVERRIDE=1`

# OneDNN
`ITEX_ONEDNN_GRAPH=1`
`_ITEX_ONEDNN_GRAPH_ALL_TYPE=1`
`TF_ENABLE_ONEDNN_OPTS=1`

# mixed precision
`ITEX_AUTO_MIXED_PRECISION=1`
* For more detailed options:
`./docs/guide/aamp_tune.md`


# for more consult 
`./docs/`
`./tests/`
