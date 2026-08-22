# Graph rewrite keys off the oneDNN library actually loaded

Native-layout extras follow the CPU oneDNN runtime this process opened in `LoadCpuLibrary`, not a second read of `ITEX_OMP_THREADPOOL`. Compile-time THREADPOOL-only builds freeze Eigen THREADPOOL and ignore `=1`. Re-reading the env at Grappler time can disagree with the `.so` already in memory.

## Considered Options

- **Re-read env + macros at rewrite time** — rejected; that is how rewrite policy drifted from the loaded runtime.
- **Treat every default Python wheel as OpenMP for rewrites** — rejected; `=0` would still load Eigen THREADPOOL oneDNN while applying OpenMP extras.
