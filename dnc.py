"""DNC Cores.

These modules create a DNC core. They take input, pass parameters to the memory
access module, and integrate the output of memory to form an output.
"""

import collections
import tensorflow as tf
import numpy as np

import memory_Access

DNCState = collections.namedtuple('DNCState',('access_output',
                                              'access_state',
                                              'controller_state'))

class DNC():
    """DNC core module.

      Contains controller and memory access module.
      """

    def __init__(self,
                 access_config, # this is a tuple
                 controller_config, # this is a tuple
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
        self.name=name

        self.controller = tf.nn.rnn_cell.BasicLSTMCell(**controller_config)
        self.access = memory_Access.Memory_Access(**access_config)
        self.output_size = output_size
        self.clip_value = clip_value or 0

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
        with tf.name_scope('dnc'):
            prev_access_output = prev_state.access_output
            prev_access_state = prev_state.access_state
            prev_controller_state = prev_state.controller_state

            batch_flatten = tf.layers.Flatten()
            controller_input = tf.concat(
                [batch_flatten(inputs), batch_flatten(prev_access_output)], 1)

            controller_output, controller_state = self.controller(
                controller_input, prev_controller_state)

            # clip
            controller_output = self.__clip_if_enabled(controller_output)
            controller_state = tf.contrib.framework.nest.map_structure(
                self.__clip_if_enabled, controller_state)

            access_output, access_state = self.access(
                controller_output, prev_access_state)

            output = tf.concat([controller_output, batch_flatten(access_output)], 1)
            output = memory_Access.linear_transform(name='output_linear',
                                                    inputs=output,
                                                    outputs_size=self.output_size)
            output = self.__clip_if_enabled(output)

            return output, DNCState(
                access_output=access_output,
                access_state=access_state,
                controller_state=controller_state)

    def __clip_if_enabled(self, x):
        if self.clip_value > 0:
            return tf.clip_by_value(x, -self.clip_value, self.clip_value)
        else:
            return x

    def initial_state(self, batch_size, dtype=tf.float32):
        return DNCState(
            controller_state=self.controller.zero_state(batch_size,dtype),
            access_state=self.access.initial_state(batch_size),
            access_output=self.access.initial_output(batch_size)
        )


if __name__ == "__main__":
    """set parameters"""
    controller_config = {
        "num_units": 4
    }
    access_config = {
        "memory_size": 10
    }
    output_size = 10
    batch_size = 3

    #############################################
    #############################################
    # test class DNC

    dnc = DNC(access_config=access_config,
              controller_config=controller_config,
              output_size=output_size)

    #########################################
    # test DNC.build
    inputs = tf.constant(np.random.rand(batch_size, 1, 5).astype(np.float32))
    prev_state = dnc.initial_state(batch_size=batch_size)
    output, dnc_state = dnc(inputs=inputs, prev_state=prev_state)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("logs/", sess.graph)
        sess.run(tf.initialize_all_variables())
        output, dnc_state = sess.run([output, dnc_state])
        print(output)
        print(dnc_state)

