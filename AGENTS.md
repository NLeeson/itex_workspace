# Repository Guidelines

## Project Structure & Module Organization
- `itex/` holds the core extension implementation (devices, graph, kernels, ops, profiler, utils) plus Python bindings under `itex/python/` and build tooling in `itex/tools/`.
- `test/` contains Python unit tests organized by area (e.g., `test/python/`, `test/sanity/`, `test/tensorflow/`).
- `docs/` covers design notes, install/build guides, and user documentation; `examples/` provides runnable demos.
- `third_party/` tracks external dependencies like onednn

## Build, Test, and Development Commands
- `./configure` sets build options (CPU/XPU, compiler paths) before Bazel builds. It regenerates `.itex_configure.bazelrc` configuration 
- `bazel build -c opt //itex/tools/pip_package:build_pip_package --verbose_failures` is the base build command we invoke with a configuration
- `bazel build -c opt --config=xpu //itex/tools/pip_package:build_pip_package --verbose_failures` builds configuration defined as xpu
- `bazel build -c opt --config=xpu //itex/tools/pip_package:build_pip_package --define=build_with_onednn_graph=true --define=build_with_graph_compiler=true --verbose_failures` builds configuration defined as xpu with the defined extras for onednn graph mode and onednn graph compiler backend
- `bazel-bin/itex/tools/pip_package/build_pip_package WHL/` packages the wheel into `WHL/`.

## Coding Style & Naming Conventions
- Follow TensorFlow style guides for Python, C++, and documentation.
- Python: use `pylint` with `.pylintrc` at repo root (`pylint --rcfile=.pylintrc myfile.py`).
- C++: format with `clang-format-12 -i -style=file <file>` and lint with `cpplint --filter=-legal/copyright --exclude=./third_party --recursive ./`.
- Bazel: format `BUILD`/`.bzl` files with `buildifier`.

## Commit & Pull Request Guidelines
- Commit subjects are short, imperative, and capitalized; optional tags like `[FIX]` appear, and PR/issue numbers are often appended in parentheses (e.g., `Fix oneDNN build (#2748)`).
- Open or reference an issue for bugs/features; significant changes require the RFC process before implementation.
- PRs should describe the change, link relevant issues, and confirm that the build and examples/tests pass when applicable.

## Context
- intel llvm sycl toolchain, icpx/icx compilers, ld.lld linker are set for the environment
- oneapi 2025.3 vars are set for the environment 
- device: gpu: adl-p; gpu family: xe-lp; environment var `AOT_CONFIG=adl-p` is set with the aot device
- `ONEAPI_DEVICE_SELECTOR=level_zero:gpu;*:cpu` environment var is set


## Configuration
- Use `docs/install/how_to_build.md` for environment prerequisites (Bazel, compiler, oneAPI) before attempting local builds.
- itex build configuration goal: itex onednn runs code on GPU via SYCL / level zero (ur runtime l0); CPU via THREADPOOL;
- `.bazelrc` bazel configuration in the build root 
- `.itex_configure.bazelrc` (generated) configuration after invoking ./configure in the build root
- `third_party/onednn/*.BUILD` files to set the configuration for oneDNN to be build with itex
