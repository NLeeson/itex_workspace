#!/usr/bin/env python3
"""Single-process runtime validation for mixed CPU pipeline + XPU training.

This script is intentionally validation-only. It proves one training step where:
- the input pipeline executes oneDNN-backed preprocessing on CPU
- the model forward/backward step executes on XPU

Run with verbose logging enabled to inspect oneDNN runtime initialization and
primitive execution from the same process.
"""

from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a single mixed CPU-pipeline + XPU-training step."
    )
    parser.add_argument("--steps", type=int, default=2, help="Training steps to execute.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--width", type=int, default=128, help="Feature width.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable ITEX and oneDNN verbose output.",
    )
    return parser.parse_args()


def configure_env(verbose: bool) -> None:
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "0")
    os.environ.setdefault("ITEX_DISABLE_XLA", "1")
    os.environ.setdefault("ITEX_ONEDNN_GRAPH", "1")
    os.environ.setdefault("_ITEX_ONEDNN_GRAPH_ALL_TYPE", "0")
    os.environ.setdefault("ITEX_OMP_THREADPOOL", "0")
    if verbose:
        os.environ["ITEX_VERBOSE"] = "1"
        os.environ["DNNL_VERBOSE"] = "1"
        os.environ["ONEDNN_VERBOSE"] = "all"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        os.environ["ITEX_CPP_MIN_LOG_LEVEL"] = "0"
    else:
        os.environ.setdefault("ITEX_VERBOSE", "0")
        os.environ.setdefault("DNNL_VERBOSE", "0")
        os.environ.setdefault("ONEDNN_VERBOSE", "0")


ARGS = parse_args()
configure_env(ARGS.verbose)

import numpy as np
import tensorflow as tf
import intel_extension_for_tensorflow as itex


def require_device(device_type: str) -> str:
    logical = tf.config.list_logical_devices(device_type)
    if not logical:
        raise RuntimeError(f"Required logical device {device_type} is not available")
    return logical[0].name


CPU_DEVICE = require_device("CPU")
XPU_DEVICE = require_device("XPU")


def build_cpu_dataset(steps: int, batch_size: int, width: int) -> tf.data.Dataset:
    total_examples = steps * batch_size

    with tf.device(CPU_DEVICE):
        cpu_w1 = tf.constant(
            np.linspace(-0.05, 0.05, width * width, dtype=np.float32).reshape(width, width)
        )
        cpu_b1 = tf.constant(np.linspace(-0.01, 0.01, width, dtype=np.float32))
        cpu_w2 = tf.constant(
            np.linspace(0.03, -0.03, width * width, dtype=np.float32).reshape(width, width)
        )
        label_w = tf.constant(
            np.linspace(0.02, -0.02, width * width, dtype=np.float32).reshape(width, width)
        )

    @tf.function
    def preprocess(ids: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        with tf.device(CPU_DEVICE):
            hot = tf.one_hot(tf.math.mod(ids, width), depth=width, dtype=tf.float32)
            hidden = tf.matmul(hot, cpu_w1)
            hidden = tf.nn.bias_add(hidden, cpu_b1)
            hidden = tf.nn.gelu(hidden, approximate=True)
            features = tf.matmul(hidden, cpu_w2)
            labels = tf.matmul(features, label_w)
            return features, labels

    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    options.experimental_optimization.apply_default_optimizations = True

    dataset = tf.data.Dataset.range(total_examples)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(preprocess, num_parallel_calls=1)
    dataset = dataset.prefetch(1)
    dataset = dataset.with_options(options)
    return dataset


def build_xpu_state(width: int) -> tuple[list[tf.Variable], tf.keras.optimizers.Optimizer]:
    with tf.device(XPU_DEVICE):
        w1 = tf.Variable(
            np.linspace(-0.04, 0.04, width * width, dtype=np.float32).reshape(width, width),
            trainable=True,
            name="xpu_w1",
        )
        b1 = tf.Variable(np.zeros((width,), dtype=np.float32), trainable=True, name="xpu_b1")
        w2 = tf.Variable(
            np.linspace(0.02, -0.02, width * width, dtype=np.float32).reshape(width, width),
            trainable=True,
            name="xpu_w2",
        )
        b2 = tf.Variable(np.zeros((width,), dtype=np.float32), trainable=True, name="xpu_b2")
        optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    return [w1, b1, w2, b2], optimizer


def main() -> int:
    tf.debugging.set_log_device_placement(bool(ARGS.verbose))

    print("TensorFlow:", tf.__version__)
    print("ITEX:", getattr(itex, "__version__", "unknown"))
    print("CPU logical device:", CPU_DEVICE)
    print("XPU logical device:", XPU_DEVICE)
    print(
        "Verbose env:",
        {
            "ITEX_VERBOSE": os.environ.get("ITEX_VERBOSE"),
            "ONEDNN_VERBOSE": os.environ.get("ONEDNN_VERBOSE"),
            "ITEX_ONEDNN_GRAPH": os.environ.get("ITEX_ONEDNN_GRAPH"),
            "ITEX_DISABLE_XLA": os.environ.get("ITEX_DISABLE_XLA"),
        },
    )

    dataset = build_cpu_dataset(ARGS.steps, ARGS.batch_size, ARGS.width)
    iterator = iter(dataset)
    variables, optimizer = build_xpu_state(ARGS.width)

    @tf.function(jit_compile=False, reduce_retracing=True)
    def train_step() -> tf.Tensor:
        features, labels = next(iterator)
        with tf.device(XPU_DEVICE):
            with tf.GradientTape() as tape:
                hidden = tf.matmul(features, variables[0]) + variables[1]
                hidden = tf.nn.gelu(hidden, approximate=True)
                logits = tf.matmul(hidden, variables[2]) + variables[3]
                loss = tf.reduce_mean(tf.math.squared_difference(logits, labels))
            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))
            return loss

    losses = []
    for step_idx in range(ARGS.steps):
        loss = float(train_step().numpy())
        if not np.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step_idx}: {loss}")
        losses.append(loss)
        print(f"step={step_idx} loss={loss:.6e}")

    print("Mixed-runtime validation completed.")
    print("Loss trajectory:", ", ".join(f"{loss:.6e}" for loss in losses))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
