# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DNC Cores.

These modules create a DNC core. They take input, pass parameters to the memory
access module, and integrate the output of memory to form an output.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf

import dnc_cell.access as access

DNCState = collections.namedtuple('DNCState', ('access_output', 'access_state',
                                               'controller_state'))


class DNC(tf.nn.rnn_cell.RNNCell):
  """DNC core module.

  Contains controller and memory access module.
  """

  def __init__(self,
               access_config,
               controller_config,
               output_size,
               clip_value=None,
               name='dnc'):
    """Initializes the DNC core.

    Args:
      access_config: dictionary of access module configurations.
      controller_config: dictionary of controller (LSTM) module configurations.
      output_size: output dimension size of core.
      clip_value: clips controller and core output values to between
          `[-clip_value, clip_value]` if specified.
      name: module name (default 'dnc').

    Raises:
      TypeError: if direct_input_size is not None for any access module other
        than KeyValueMemory.
    """
    super(DNC, self).__init__(name=name)

    with tf.name_scope("controller_and_access"):
      self._controller = tf.nn.rnn_cell.BasicLSTMCell(**controller_config)
      self._access = access.MemoryAccess(**access_config)

    self._access_output_size = np.prod(self._access.output_size.as_list())
    self._output_size = output_size
    self._clip_value = clip_value or 0

    self._output_size = tf.TensorShape([output_size])
    self._state_size = DNCState(
        access_output=self._access_output_size,
        access_state=self._access.state_size,
        controller_state=self._controller.state_size)

  def _clip_if_enabled(self, x):
    if self._clip_value > 0:
      return tf.clip_by_value(x, -self._clip_value, self._clip_value)
    else:
      return x

  def __call__(self, inputs, prev_state):
      return self._build(inputs, prev_state)

  def _build(self, inputs, prev_state):
    """Connects the DNC core into the graph.

    Args:
      inputs: Tensor input.
      prev_state: A `DNCState` tuple containing the fields `access_output`,
          `access_state` and `controller_state`. `access_state` is a 3-D Tensor
          of shape `[batch_size, num_reads, word_size]` containing read words.
          `access_state` is a tuple of the access module's state, and
          `controller_state` is a tuple of controller module's state.

    Returns:
      A tuple `(output, next_state)` where `output` is a tensor and `next_state`
      is a `DNCState` tuple containing the fields `access_output`,
      `access_state`, and `controller_state`.
    """

    prev_access_output = prev_state.access_output
    prev_access_state = prev_state.access_state
    prev_controller_state = prev_state.controller_state

    batch_flatten = tf.layers.Flatten()
    controller_input = tf.concat(
        [batch_flatten(inputs), batch_flatten(prev_access_output)], 1)

    # the shape of controller_output is `[batch_size, num_units]`
    controller_output, controller_state = self._controller(
        controller_input, prev_controller_state)

    controller_output = self._clip_if_enabled(controller_output)

    controller_state = tf.contrib.framework.nest.map_structure(
        self._clip_if_enabled, controller_state)

    access_output, access_state = self._access(controller_output,
                                               prev_access_state)

    output = tf.concat([controller_output, batch_flatten(access_output)], 1)
    # output1 = snt.Linear(
    #     output_size=self._output_size.as_list()[0],
    #     name='output_linear')(output)
    # print(output1)
    output = access.linear_transform(
            name='dnc_linear',
            inputs=output,
            outputs_size=self._output_size.as_list()[0])
    # print(output)
    output = self._clip_if_enabled(output)

    return output, DNCState(
        access_output=access_output,
        access_state=access_state,
        controller_state=controller_state)

  def initial_state(self, batch_size, dtype=tf.float32):
    return DNCState(
        controller_state=self._controller.zero_state(batch_size,dtype),
        access_state=self._access.zero_state(batch_size, dtype),
        access_output=tf.zeros(
            [batch_size] + self._access.output_size.as_list(), dtype)
    )

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size
