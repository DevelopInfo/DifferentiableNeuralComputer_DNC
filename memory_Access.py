import tensorflow as tf
import collections

AccessState = collections.namedtuple('AccessState', (
    'memory', 'read_weights', 'write_weights', 'linkage', 'usage'))

class Memory_Access():
    def __init__(self,
                 memory_size=128,
                 word_size=20,
                 num_reads=2,
                 num_writes=1,
                 name='memory_access'):
        self.name = name
        self.memory_size = memory_size
        self.word_size = word_size
        self.num_reads = num_reads
        self.num_writes = num_writes

    def build(self, inputs, prev_access_state=None):
        """
        Args:
          inputs: tensor of shape = [batch_size, input_size]. This is used to
              control this access module.
          prev_access_state: Instance of `AccessState` containing the previous state.

        Returns:
          A tuple = (output, next_state), where `output` is a tensor of shape =
          [batch_size, num_reads, word_size], and `next_state` is the new
          `AccessState` named tuple at the current time t.
        """
        inputs = self.parse_input(controller_input=inputs)

    def parse_input(self, controller_input):
        """parse the controller_input
        Args:
            controller_input: tensor of shape = [batch_size, WR+3W+5R+3].
        Returns:
            A tuple = (read_keys, read_strengths, write_key, write_strength,
            erase_vector, write_vector, free_gates, allocation_gate,
            write_gate, read_modes).
        """
        # print(controller_input)

        def oneplus(x):
            return 1 + tf.log( 1 + tf.exp(x))

        read_keys = controller_input[:,
                    0:self.num_reads*self.word_size]

        read_strengths = controller_input[:,
                         self.num_reads*self.word_size :
                         self.num_reads*self.word_size+self.num_reads]
        read_strengths = oneplus(read_strengths)

        write_key = controller_input[:,
                    self.num_reads*self.word_size+self.num_reads :
                    self.num_reads*self.word_size+self.num_reads+self.word_size]

        write_strength = controller_input[:,
                         self.num_reads*self.word_size+self.num_reads+self.word_size :
                         self.num_reads*self.word_size+self.num_reads+self.word_size + 1]
        write_strength = oneplus(write_strength)

        erase_vector =  controller_input[:,
                        self.num_reads * self.word_size + self.num_reads + self.word_size + 1 :
                        self.num_reads * self.word_size + self.num_reads + 2 * self.word_size + 1]
        erase_vector = tf.sigmoid(erase_vector)

        write_vector = controller_input[:,
                       self.num_reads * self.word_size + self.num_reads + 2 * self.word_size + 1:
                       self.num_reads * self.word_size + self.num_reads + 3 * self.word_size + 1]

        free_gates = controller_input[:,
                     self.num_reads * self.word_size + self.num_reads + 3 * self.word_size + 1:
                     self.num_reads * self.word_size + 2 * self.num_reads + 3 * self.word_size + 1]
        free_gates = tf.sigmoid(free_gates)

        allocation_gate = controller_input[:,
                          self.num_reads * self.word_size + 2 * self.num_reads + 3 * self.word_size + 1:
                          self.num_reads * self.word_size + 2 * self.num_reads + 3 * self.word_size + 2]
        allocation_gate = tf.sigmoid(allocation_gate)

        write_gate = controller_input[:,
                     self.num_reads * self.word_size + 2 * self.num_reads + 3 * self.word_size + 2:
                     self.num_reads * self.word_size + 2 * self.num_reads + 3 * self.word_size + 3]
        write_gate = tf.sigmoid(write_gate)

        read_modes = controller_input[:,
                     self.num_reads * self.word_size + 2 * self.num_reads + 3 * self.word_size + 3:
                     self.num_reads * self.word_size + 5 * self.num_reads + 3 * self.word_size + 4]
        read_modes = tf.nn.softmax(read_modes)

        result = {
            'read_keys': read_keys,
            'read_strengths': read_strengths,
            'write_key': write_key,
            'write_strength': write_strength,
            'write_vector': write_vector,
            'erase_vector': erase_vector,
            'free_gates': free_gates,
            'allocation_gate': allocation_gate,
            'write_gate': write_gate,
            'read_modes': read_modes,
        }
        # print(result)

        return result


# test

import numpy as np

if __name__ == "__main__":
    """set parameters"""
    batch_size = 3
    memory_size = 128
    word_size = 20
    num_reads = 2
    num_writes = 1

    controller_input = tf.constant(np.ones(shape=(batch_size, word_size*num_reads + 3*word_size + 5*num_reads + 3)))

    memoryAccess = Memory_Access(memory_size, word_size, num_reads, num_writes)
    result = memoryAccess.parse_input(controller_input)

    with tf.Session() as sess:
        result_value = sess.run(result)
        print(result_value['read_modes'])
