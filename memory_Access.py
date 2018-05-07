import tensorflow as tf
import collections
import memory_Addressing
import numpy as np

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

        self.write_content_weights_mod = memory_Addressing.CosineWeights(
            num_heads=num_writes,
            word_size=word_size,
            name='write_content_weights'
        )
        self.read_content_weights_mod = memory_Addressing.CosineWeights(
            num_heads=num_reads,
            word_size=word_size,
            name='read_content_weights'
        )

        self.allocation = memory_Addressing.Allocation(
            memory_size=memory_size
        )

        self.linkage = memory_Addressing.TemporalLinkage(
            memory_size=memory_size,
            num_writes=num_writes
        )


    def build(self, inputs, prev_access_state):
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

        # Update usage using inputs['free_gate'] and previous read & write weights.
        usage = self.allocation.get_usage(
            write_weights=prev_access_state.write_weights,
            free_gates=inputs['free_gates'],
            read_weights=prev_access_state.read_weights,
            prev_usage=prev_access_state.usage)

        # write to memory
        write_weights = self.write_weights(
            inputs=inputs,
            memory=prev_access_state.memory,
            usage=usage
        )
        memory = self.erase_and_write(
            memory=prev_access_state.memory,
            address=write_weights,
            reset_weights=inputs['erase_vector'],
            values=inputs['write_vector']
        )
        linkage_state = self.linkage.build(
            write_weights=write_weights,
            prev_state=prev_access_state.linkage
        )

        # read from memory
        read_weights = self.read_weights(
            inputs=inputs,
            memory=memory,
            prev_read_weights=prev_access_state.read_weights,
            link=linkage_state.link
        )
        # read_words.shape = [batch_size, num_reads, word_size]
        read_words = tf.matmul(read_weights, memory)

        return read_words, AccessState(
            memory=memory,
            read_weights=read_weights,
            write_weights=write_weights,
            linkage=linkage_state,
            usage=usage
        )

    def parse_input(self, controller_input):
        """parse the controller_input
        Args:
            controller_input: tensor of shape = [batch_size, input_size].
        Returns:
            A tuple = (read_keys, read_strengths, write_key, write_strength,
            erase_vector, write_vector, free_gates, allocation_gate,
            write_gate, read_modes).
        """
        # expand dimension
        inputs = tf.expand_dims(input=controller_input,
                                axis=1)
        # linear transformation
        inputs = tf.matmul(inputs,
                           tf.Variable(tf.random_normal(shape=[inputs.get_shape().as_list()[0],
                                                               inputs.get_shape().as_list()[2],
                                                               self.word_size * self.num_reads + 3 * self.word_size + 5 * self.num_reads + 3],
                                                        dtype=tf.float32)
                                       ))
        # decrease dimension
        controller_input = tf.reduce_sum(input_tensor=inputs,
                               axis=1)


        def oneplus(x):
            return 1 + tf.log( 1 + tf.exp(x))

        batch_size = controller_input.shape[0]

        # read_keys.shape = (batch_size, num_reads, word_size)
        read_keys = controller_input[:,
                    0:self.num_reads*self.word_size]
        read_keys = tf.reshape(
            tensor=read_keys,
            shape=(batch_size, self.num_reads, self.word_size))

        # read_strength.shape = (batch_size, num_reads)
        read_strengths = controller_input[:,
                         self.num_reads*self.word_size :
                         self.num_reads*self.word_size+self.num_reads]
        read_strengths = oneplus(read_strengths)

        # write_key.shape = (batch_size, num_writes, word_size)
        write_key = controller_input[:,
                    self.num_reads*self.word_size+self.num_reads :
                    self.num_reads*self.word_size+self.num_reads+self.word_size]
        write_key = tf.reshape(
            tensor=write_key,
            shape=(batch_size, self.num_writes, self.word_size)
        )

        # write_strength.shape = (batch_size, num_writes)
        write_strength = controller_input[:,
                         self.num_reads*self.word_size+self.num_reads+self.word_size :
                         self.num_reads*self.word_size+self.num_reads+self.word_size + 1]
        write_strength = oneplus(write_strength)

        # erase_vector.shape = (batch_size, num_writes, word_size)
        erase_vector =  controller_input[:,
                        self.num_reads * self.word_size + self.num_reads + self.word_size + 1 :
                        self.num_reads * self.word_size + self.num_reads + 2 * self.word_size + 1]
        erase_vector = tf.sigmoid(erase_vector)
        erase_vector = tf.reshape(
            tensor=erase_vector,
            shape=(batch_size, self.num_writes, self.word_size)
        )

        # write_vector.shape = (batch_size, num_writes, word_size)
        write_vector = controller_input[:,
                       self.num_reads * self.word_size + self.num_reads + 2 * self.word_size + 1:
                       self.num_reads * self.word_size + self.num_reads + 3 * self.word_size + 1]
        write_vector = tf.reshape(
            tensor=write_vector,
            shape=(batch_size, self.num_writes, self.word_size)
        )

        # free_gates.shape = (batch_size, num_reads)
        free_gates = controller_input[:,
                     self.num_reads * self.word_size + self.num_reads + 3 * self.word_size + 1:
                     self.num_reads * self.word_size + 2 * self.num_reads + 3 * self.word_size + 1]
        free_gates = tf.sigmoid(free_gates)

        # allocation_gate.shape = (batch_size, num_writes)
        allocation_gate = controller_input[:,
                          self.num_reads * self.word_size + 2 * self.num_reads + 3 * self.word_size + 1:
                          self.num_reads * self.word_size + 2 * self.num_reads + 3 * self.word_size + 2]
        allocation_gate = tf.sigmoid(allocation_gate)

        # write_gate.shape = (batch_size, num_writes)
        write_gate = controller_input[:,
                     self.num_reads * self.word_size + 2 * self.num_reads + 3 * self.word_size + 2:
                     self.num_reads * self.word_size + 2 * self.num_reads + 3 * self.word_size + 3]
        write_gate = tf.sigmoid(write_gate)

        # read_modes.shape = (batch_size, num_reads, 3)
        read_modes = controller_input[:,
                     self.num_reads * self.word_size + 2 * self.num_reads + 3 * self.word_size + 3:
                     self.num_reads * self.word_size + 5 * self.num_reads + 3 * self.word_size + 4]

        read_modes = tf.reshape(
            tensor=read_modes,
            shape=(batch_size, self.num_reads, 3)
        )
        read_modes = tf.nn.softmax(logits=read_modes, axis=2)

        result = {
            # [batch_size, num_reads, word_size]
            'read_keys': read_keys,
            # [batch_size, num_reads]
            'read_strengths': read_strengths,
            # [batch_size, num_writes, word_size]
            'write_key': write_key,
            # [batch_size, num_writes]
            'write_strength': write_strength,
            # [batch_size, num_writes, word_size]
            'write_vector': write_vector,
            # [batch_size, num_writes, word_size]
            'erase_vector': erase_vector,
            # [batch_size, num_reads]
            'free_gates': free_gates,
            # [batch_size, num_wirtes]
            'allocation_gate': allocation_gate,
            # [batch_size, num_writes]
            'write_gate': write_gate,
            # [batch_size, num_reads, 3]
            'read_modes': read_modes,
        }

        return result

    def erase_and_write(self, memory, address, reset_weights, values):
        """Module to erase and write in the external memory.

          Erase operation:
            M_t'(i) = M_{t-1}(i) * (1 - w_t(i) * e_t)

          Add operation:
            M_t(i) = M_t'(i) + w_t(i) * a_t

          where e are the reset_weights, w the write weights and a the values.

          Args:
            memory: 3-D tensor of shape `[batch_size, memory_size, word_size]`.
            address: 3-D tensor `[batch_size, num_writes, memory_size]`.
            reset_weights: 3-D tensor `[batch_size, num_writes, word_size]`.
            values: 3-D tensor `[batch_size, num_writes, word_size]`.

          Returns:
            3-D tensor of shape `[batch_size, num_writes, word_size]`.
          """
        with tf.name_scope('erase_memory', values=[memory, address, reset_weights]):
            expand_address = tf.expand_dims(address, 3)
            reset_weights = tf.expand_dims(reset_weights, 2)
            weighted_resets = expand_address * reset_weights
            reset_gate = tf.reduce_prod(1 - weighted_resets, [1])
            memory *= reset_gate

        with tf.name_scope('additive_write', values=[memory, address, values]):
            add_matrix = tf.matmul(address, values, adjoint_a=True)
            memory += add_matrix

        return memory

    def write_weights(self, inputs, memory, usage):
        """Calculates the memory locations to write to.

        This uses a combination of content-based lookup and finding an unused
        location in memory, for each write head.

        Args:
          inputs: Collection of inputs to the access module, including controls for
              how to chose memory writing, such as the content to look-up and the
              weighting between content-based and allocation-based addressing.
          memory: A tensor of shape  `[batch_size, memory_size, word_size]`
              containing the current memory contents.
          usage: Current memory usage, which is a tensor of shape `[batch_size,
              memory_size]`, used for allocation-based addressing.

        Returns:
          tensor of shape `[batch_size, num_writes, memory_size]` indicating where
              to write to (if anywhere) for each write head.
        """
        with tf.name_scope('write_weights', values=[inputs, memory, usage]):
            # c_t^{w, i} - The content-based weights for each write head.
            write_content_weights = self.write_content_weights_mod.build(
                memory=memory,
                keys=inputs['write_key'],
                strengths=inputs['write_strength']
            )

            # a_t^i - The allocation weights for each write head.
            write_allocation_weights = self.allocation.write_allocation_weights(
                usage=usage,
                write_gates=(inputs['allocation_gate'] * inputs['write_gate']),
                num_writes=self.num_writes
            )

            # Expands gates over memory locations.
            allocation_gate = tf.expand_dims(inputs['allocation_gate'], -1)
            write_gate = tf.expand_dims(inputs['write_gate'], -1)

            # w_t^{w, i} - The write weightings for each write head.
            return write_gate * (allocation_gate * write_allocation_weights +
                                 (1 - allocation_gate) * write_content_weights)

    def read_weights(self, inputs, memory, prev_read_weights, link):
        """Calculates read weights for each read head.

        The read weights are a combination of following the link graphs in the
        forward or backward directions from the previous read position, and doing
        content-based lookup. The interpolation between these different modes is
        done by `inputs['read_mode']`.

        Args:
          inputs: Controls for this access module. This contains the content-based
              keys to lookup, and the weightings for the different read modes.
          memory: A tensor of shape `[batch_size, memory_size, word_size]`
              containing the current memory contents to do content-based lookup.
          prev_read_weights: A tensor of shape `[batch_size, num_reads,
              memory_size]` containing the previous read locations.
          link: A tensor of shape `[batch_size, num_writes, memory_size,
              memory_size]` containing the temporal write transition graphs.

        Returns:
          A tensor of shape `[batch_size, num_reads, memory_size]` containing the
          read weights for each read head.
        """
        with tf.name_scope(
                'read_weights', values=[inputs, memory, prev_read_weights, link]):
            # c_t^{r, i} - The content weightings for each read head.
            content_weights = self.read_content_weights_mod.build(
                memory, inputs['read_keys'], inputs['read_strengths'])

            # Calculates f_t^i and b_t^i.
            forward_weights = self.linkage.directional_read_weights(
                link, prev_read_weights, forward=True)
            backward_weights = self.linkage.directional_read_weights(
                link, prev_read_weights, forward=False)

            backward_mode = inputs['read_modes'][:, :, 0:1]
            forward_mode = (
                inputs['read_modes'][:, :,  2:3])
            content_mode = inputs['read_modes'][:, :, 1:2]

            # content_weights.shape = (batch_size, num_reads, memory_size)
            # forward_weights.shape = (batch_size, num_reads, num_writes, memory_size)
            # backward_weights.shape = (bach_size, num_reads, num_writes, memory_size)
            # read_weights.shape = (batch_size, num_reads, memory_size)
            read_weights = (
                    content_mode * content_weights +
                    tf.reduce_sum(tf.expand_dims(forward_mode, 3) * forward_weights, 2) +
                    tf.reduce_sum(tf.expand_dims(backward_mode, 3) * backward_weights, 2))

            return read_weights

    def initial_state(self, batch_size, dtype=tf.float64):
        AccessState.memory = tf.constant(value=10 * (np.random.rand(batch_size, self.memory_size, self.word_size) - 0.5),
                                         dtype=dtype)

        AccessState.read_weights = tf.constant(value=np.random.rand(batch_size, self.num_reads, self.memory_size),
                                         dtype=dtype)

        AccessState.write_weights = tf.constant(value=np.random.rand(batch_size, self.num_writes, self.memory_size),
                                         dtype=dtype)

        memory_Addressing.TemporalLinkageState.link = tf.constant(
            value=np.random.rand(batch_size, self.num_writes, self.memory_size, self.memory_size),
                                         dtype=dtype)
        memory_Addressing.TemporalLinkageState.precedence_weights = tf.constant(
            value=np.random.rand(batch_size, self.num_writes, self.memory_size),
                                         dtype=dtype)
        AccessState.linkage = memory_Addressing.TemporalLinkageState

        AccessState.usage = tf.constant(value=np.random.rand(batch_size, self.memory_size),
                                         dtype=dtype)

        return AccessState

    def initial_output(self, batch_size, dtype=tf.float64):
        return tf.constant(value=np.random.rand(batch_size, self.num_reads, self.word_size),
                           dtype=dtype)

# test

if __name__ == "__main__":
    """set parameters"""
    batch_size = 3
    memory_size = 128
    word_size = 20
    num_reads = 2
    num_writes = 1

    ###############################################
    ###############################################
    # test class Memeory_Access

    memoryAccess = Memory_Access(memory_size, word_size, num_reads, num_writes)

    ################################################
    # test Memeory_Access.parse_input

    # controller_input = tf.random_normal(
    #     [batch_size,
    #     word_size*num_reads + 3*word_size + 5*num_reads + 3])
    #
    # result = memoryAccess.parse_input(controller_input)
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     result_value = sess.run(result)
    #     print(result_value['read_modes'])
    #     print(result_value['read_modes'].sum(2))

    ####################################################
    # test Memory_Access.write_weights

    controller_input = tf.random_normal(
            shape=[batch_size,
            word_size*num_reads + 3*word_size + 5*num_reads + 3],
            dtype=tf.float64)

    input = memoryAccess.parse_input(controller_input)

    memory = tf.constant(np.random.rand(batch_size, memory_size, word_size))

    usage = tf.constant(np.random.rand(batch_size, memory_size))
    write_weights = memoryAccess.write_weights(inputs=input,
                                               memory=memory,
                                               usage=usage)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        write_weights = sess.run(write_weights)
        print(write_weights)

    ######################################################
    # test Memory_Access.read_weights

    # controller_input = tf.constant(
    #     np.random.rand(batch_size, word_size * num_reads + 3 * word_size + 5 * num_reads + 3))
    # input = memoryAccess.parse_input(controller_input)
    #
    # memory = tf.constant(np.random.rand(batch_size, memory_size, word_size))
    #
    # prev_read_weights = tf.constant(np.random.rand(batch_size, num_reads, memory_size))
    #
    # link = tf.constant(np.random.rand(batch_size, num_writes, memory_size, memory_size))
    # read_weights = memoryAccess.read_weights(inputs=input,
    #                                           memory=memory,
    #                                           prev_read_weights=prev_read_weights,
    #                                           link=link)
    #
    # with tf.Session() as sess:
    #     read_weights = sess.run(read_weights)
    #     print(read_weights)

    ########################################################
    # test Memory_Access.build

    # controller_input = tf.constant(
    #     np.random.rand(batch_size, word_size * num_reads + 3 * word_size + 5 * num_reads + 3))
    #
    # AccessState = memoryAccess.initial_state(batch_size=batch_size)
    #
    # read_words, access_state = memoryAccess.build(inputs=controller_input,
    #                                               prev_access_state=AccessState)
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     read_words, access_state = sess.run([read_words, access_state])
    #     print("read_words: \n", read_words.shape)
    #     print("access_state: \n", access_state)