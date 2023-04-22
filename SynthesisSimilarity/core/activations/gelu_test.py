# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for the Gaussian error linear unit."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


from . import gelu


def test_gelu():
  expected_data = [[0.14967535, 0., -0.10032465],
                   [-0.15880796, -0.04540223, 2.9963627]]
  gelu_data = gelu([[.25, 0, -.25], [-1, -2, 3]])
  # assertAllClose(expected_data, gelu_data)


if __name__ == '__main__':
  test_gelu()
