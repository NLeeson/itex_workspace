# Native-layout extras are coupled to the OpenMP CPU oneDNN runtime

Graph rewrite policy may depend on the selected CPU oneDNN runtime. The OpenMP native-layout extras are restored when that runtime is OpenMP and stay off for Eigen THREADPOOL, matching `intel/main`’s `=0` / threadpool-build path. A single shared table (this branch today) is a default-runtime regression, not unification.

OpenMP work on this path is capability restore, not putting ITEX Eigen kernels on the OpenMP team. Eigen THREADPOOL work is adapter fidelity to submit-onto (stock TF / oneDNN `threadpool_iface`), not a deeper scheduler.

## Considered Options

- **One rewrite table for both runtimes** — rejected; it drops the extras on default OpenMP.
- **Two fully independent tables** — not chosen; no THREADPOOL-only table is required yet.
- **Put ITEX Eigen compute on the OpenMP team** — rejected for this work; that is a new OpenMP product, not O1.
