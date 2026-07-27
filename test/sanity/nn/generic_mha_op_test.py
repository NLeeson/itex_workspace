"""Correctness tests for generic float32 XPU scaled-dot-product attention."""

import numpy as np
import tensorflow as tf

from intel_extension_for_tensorflow.python.ops.multi_head_attention import (
    scaled_dot_product_attention,
)
from intel_extension_for_tensorflow.python.test_func import test
from intel_extension_for_tensorflow.python.test_func import test_util


class GenericMhaOpTest(test_util.TensorFlowTestCase):

  def _run_attention(self, query, key, value, mask, dropout, *, fast):
    query = tf.Variable(query)
    key = tf.Variable(key)
    value = tf.Variable(value)
    with tf.GradientTape() as tape:
      output = scaled_dot_product_attention(
          query,
          key,
          value,
          atten_mask=mask,
          dropout_p=dropout,
          seed=(17, 29),
          use_fast_attention=fast,
          is_training=True,
      )
      loss = tf.reduce_sum(tf.square(output))
    gradients = tape.gradient(loss, (query, key, value))
    return output, gradients

  def test_forward_and_backward_match_portable_attention(self):
    if not tf.config.list_logical_devices("XPU"):
      self.skipTest("generic attention kernel requires an XPU")

    generator = np.random.default_rng(1234)
    query = generator.normal(size=(2, 4, 7, 4)).astype(np.float32)
    key = generator.normal(size=(2, 4, 5, 4)).astype(np.float32)
    value = generator.normal(size=(2, 4, 5, 4)).astype(np.float32)
    mask = np.zeros((2, 1, 7, 5), dtype=np.float32)
    mask[:, :, :, -1] = -1.0e9

    for dropout in (0.0, 0.25):
      expected_output, expected_gradients = self._run_attention(
          query, key, value, mask, dropout, fast=False
      )
      actual_output, actual_gradients = self._run_attention(
          query, key, value, mask, dropout, fast=True
      )
      self.assertAllClose(actual_output, expected_output, rtol=2e-4, atol=2e-4)
      for actual, expected in zip(actual_gradients, expected_gradients):
        self.assertAllClose(actual, expected, rtol=3e-4, atol=3e-4)


if __name__ == "__main__":
  test.main()
