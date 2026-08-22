# Dual CPU oneDNN runtimes; Eigen THREADPOOL submits onto TensorFlow intra-op

ITEX ships two CPU oneDNN runtimes (OpenMP and Eigen THREADPOOL) and will not judge one by the other’s contract. Eigen THREADPOOL oneDNN is specified to run on an existing framework pool: oneDNN submits `parallel_for` through a threadpool adapter onto TensorFlow’s intra-op pool. TensorFlow does not schedule through oneDNN. Reverse scheduling is a different design, not this runtime’s spec. oneDNN Graph is a compute partitioner, not a worker owner.

## Considered Options

- **Treat THREADPOOL as “TF schedules through oneDNN”** — rejected; that inverts the documented `threadpool_iface` call direction.
- **One runtime only** — rejected; the default Python wheel still selects OpenMP or Eigen THREADPOOL at load, and the compile-time freeze is a third artifact, not a replacement story.
- **Call the adapter a unified threadpool** — rejected; the pool is TensorFlow’s; the adapter is only the iface.

## Consequences

OpenMP improvements (affinity, layout rewrites, inter-op throttling) must not be justified from THREADPOOL docs. Eigen THREADPOOL improvements are adapter fidelity to that submit-onto contract, not a new scheduler.
