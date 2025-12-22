# Repository Guidelines

## Project Structure & Module Organization
- `itex/` holds the core extension implementation (devices, graph, kernels, ops, profiler, utils) plus Python bindings under `itex/python/` and build tooling in `itex/tools/`.
- `test/` contains Python unit tests organized by area (e.g., `test/python/`, `test/sanity/`, `test/tensorflow/`).
- `docs/` covers design notes, install/build guides, and user documentation; `examples/` provides runnable demos.
- `third_party/` and `third-party-programs/` track external dependencies and licenses.

## Build, Test, and Development Commands
- `./configure` sets build options (CPU/XPU, compiler paths) before Bazel builds.
- `bazel build -c opt --config=cpu //itex/tools/pip_package:build_pip_package` builds a CPU wheel artifact.
- `bazel build -c opt --config=xpu //itex/tools/pip_package:build_pip_package` builds an XPU wheel artifact.
- `bazel-bin/itex/tools/pip_package/build_pip_package WHL/` packages the wheel into `WHL/`.
- `python <path_to_test.py>` runs a single Python unit test.
- `for ut in $(find test -name "*.py"); do python $ut; done` runs the full Python test suite.

## Coding Style & Naming Conventions
- Follow TensorFlow style guides for Python, C++, and documentation.
- Python: use `pylint` with `.pylintrc` at repo root (`pylint --rcfile=.pylintrc myfile.py`).
- C++: format with `clang-format-12 -i -style=file <file>` and lint with `cpplint --filter=-legal/copyright --exclude=./third_party --recursive ./`.
- Bazel: format `BUILD`/`.bzl` files with `buildifier`.

## Testing Guidelines
- Tests are Python-first; keep new tests under `test/` with clear, descriptive filenames (match existing folder conventions).
- Validate fixes with targeted tests and run the full suite for broad changes.

## Commit & Pull Request Guidelines
- Commit subjects are short, imperative, and capitalized; optional tags like `[FIX]` appear, and PR/issue numbers are often appended in parentheses (e.g., `Fix oneDNN build (#2748)`).
- Open or reference an issue for bugs/features; significant changes require the RFC process before implementation.
- PRs should describe the change, link relevant issues, and confirm that the build and examples/tests pass when applicable.

## Security & Configuration Tips
- Review `SECURITY.md` for vulnerability reporting guidance.
- Use `docs/install/how_to_build.md` for environment prerequisites (Bazel, compiler, oneAPI) before attempting local builds.
