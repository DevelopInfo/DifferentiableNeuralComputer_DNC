import tensorflow as tf
import collections
import memory_Addressing
import numpy as np


def _linear(name, inputs, first_dim=1, second_dim=1, bias_enable=False, activation=None):
    """Returns a linear transformation of `inputs`, followed by a reshape."""
    output_size = first_dim * second_dim
    # expand dimension
    inputs = tf.expand_dims(input=inputs, axis=1)

    # linear transformation
    with tf.variable_scope(name+"_weightscope",reuse=tf.AUTO_REUSE):
        weights = tf.get_variable(
            name= name + "_weights",
            shape=(inputs.get_shape().as_list()[0],
                   inputs.get_shape().as_list()[2],
                   output_size),
            dtype=tf.float32)
    linear = tf.matmul(inputs, weights)

    if bias_enable:
        bias = tf.get_variable(
            name = name + "_bias",
            shape=(inputs.get_shape().as_list()[0],1, output_size),
            dtype=tf.float32)
        inputs += bias

    if activation is not None:
        linear = activation(linear)
    return tf.reshape(linear, [-1, first_dim, second_dim])


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

    def __call__(self, inputs, prev_access_state):
        return self._build(inputs, prev_access_state)

    def _build(self, inputs, prev_access_state):
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
        with tf.name_scope("memory_access"):
            inputs = self._parse_input(controller_input=inputs)

            # Update usage using inputs['free_gates'] and previous read & write weights.
            usage = self.allocation(
                write_weights=prev_access_state.write_weights,
                free_gates=inputs['free_gates'],
                read_weights=prev_access_state.read_weights,
                prev_usage=prev_access_state.usage)

            # write to memory
            write_weights = self._write_weights(
                inputs=inputs,
                memory=prev_access_state.memory,
                usage=usage
            )
            memory = self._erase_and_write(
                memory=prev_access_state.memory,
                address=write_weights,
                reset_weights=inputs['erase_vector'],
                values=inputs['write_vector']
            )
            linkage_state = self.linkage(
                write_weights=write_weights,
                prev_state=prev_access_state.linkage
            )

            # read from memory
            read_weights = self._read_weights(
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

    def _parse_input(self, controller_input):
        """parse the controller_input
        Args:
            controller_input: tensor of shape = [batch_size, input_size] and dtype = tf.float64
        Returns:
            A tuple = (read_keys, read_strengths, write_key, write_strength,
            erase_vector, write_vector, free_gates, allocation_gate,
            write_gate, read_modes).
        """
        with tf.name_scope("parse_input"):

            def oneplus(x):
                with tf.name_scope('oneplus'):
                    return 1 + tf.log( 1 + tf.exp(x))

            # v_t^i - The vectors to write to memory, for each write head `i`.
            write_vectors = _linear(name='write_vectors',
                                    first_dim=self.num_writes,
                                    second_dim=self.word_size,
                                    inputs=controller_input)

            # e_t^i - Amount to erase the memory by before writing, for each write head.
            erase_vectors = _linear(name='erase_vectors',
                                    first_dim=self.num_writes,
                                    second_dim=self.word_size,
                                    inputs=controller_input,
                                    activation=tf.sigmoid)

            # f_t^j - Amount that the memory at the locations read from at the previous
            # time step can be declared unused, for each read head `j`.
            free_gate = tf.sigmoid(
                tf.reduce_sum(input_tensor=_linear(name='free_gate', first_dim=self.num_reads, inputs=controller_input),
                              axis=2))

            # g_t^{a, i} - Interpolation between writing to unallocated memory and
            # content-based lookup, for each write head `i`. Note: `a` is simply used to
            # identify this gate with allocation vs writing (as defined below).
            allocation_gate = tf.sigmoid(
                tf.reduce_sum(
                    input_tensor=_linear(name='allocation_gate', first_dim=self.num_writes, inputs=controller_input),
                    axis=2))

            # g_t^{w, i} - Overall gating of write amount for each write head.
            write_gate = tf.sigmoid(
                tf.reduce_sum(input_tensor=_linear(name='write_gate', first_dim=self.num_writes, inputs=controller_input),
                              axis=2))

            # \pi_t^j - Mixing between "backwards" and "forwards" positions (for
            # each write head), and content-based lookup, for each read head.
            num_read_modes = 1 + 2 * self.num_writes
            read_mode = tf.nn.softmax(logits=_linear(name='read_mode',
                                                     first_dim=self.num_reads,
                                                     second_dim=num_read_modes,
                                                     inputs=controller_input),
                                      axis=-1)

            # Parameters for the (read / write) "weights by content matching" modules.
            write_keys = _linear(name='write_keys',
                                 first_dim=self.num_writes,
                                 second_dim=self.word_size,
                                 inputs=controller_input)
            write_strengths = tf.reduce_sum(
                input_tensor=_linear(name='write_strength', first_dim=self.num_writes, inputs=controller_input),
                axis=2)

            read_keys = _linear(name='read_keys',
                                first_dim=self.num_reads,
                                second_dim=self.word_size,
                                inputs=controller_input)
            read_strengths = tf.reduce_sum(
                input_tensor=_linear(name='read_strengths', first_dim=self.num_reads, inputs=controller_input),
                axis=2)

            result = {
                # [batch_size, num_reads, word_size]
                'read_keys': read_keys,
                # [batch_size, num_reads]
                'read_strengths': read_strengths,
                # [batch_size, num_writes, word_size]
                'write_key': write_keys,
                # [batch_size, num_writes]
                'write_strength': write_strengths,
                # [batch_size, num_writes, word_size]
                'write_vector': write_vectors,
                # [batch_size, num_writes, word_size]
                'erase_vector': erase_vectors,
                # [batch_size, num_reads]
                'free_gates': free_gate,
                # [batch_size, num_wirtes]
                'allocation_gate': allocation_gate,
                # [batch_size, num_writes]
                'write_gate': write_gate,
                # [batch_size, num_reads, 3]
                'read_modes': read_mode,
            }

            return result

    def _erase_and_write(self, memory, address, reset_weights, values):
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
        with tf.name_scope('erase_and_write'):
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

    def _write_weights(self, inputs, memory, usage):
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
            write_content_weights = self.write_content_weights_mod(
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

    def _read_weights(self, inputs, memory, prev_read_weights, link):
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
            content_weights = self.read_content_weights_mod(
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

    def initial_state(self, batch_size, dtype=tf.float32):
        with tf.name_scope('initial_state'):
            # initial_memory
            initial_memory = 10 * (np.random.rand(batch_size,
                                                  self.memory_size,
                                                  self.word_size).astype(np.float32) - 0.5)

            # initial_read_weights
            initial_read_weights = np.random.rand(batch_size,
                                                  self.num_reads,
                                                  self.memory_size).astype(np.float32)
            initial_read_weights /= initial_read_weights.sum(axis=2, keepdims=True) + 1

            # initial_write_weights
            initial_write_weights = np.random.rand(batch_size,
                                                   self.num_writes,
                                                   self.memory_size).astype(np.float32)
            initial_write_weights /= initial_write_weights.sum(axis=2, keepdims=True) + 1

            # initial_linkage
            #   initial_link
            initial_link = np.random.rand(batch_size,
                                          self.num_writes,
                                          self.memory_size,
                                          self.memory_size).astype(np.float32)
            #       Row and column sums should be at most 1:
            initial_link /= np.maximum(initial_link.sum(2, keepdims=True), 1)
            initial_link /= np.maximum(initial_link.sum(3, keepdims=True), 1)

            #   initial_precedence_weights
            initial_precedence_weights = np.random.rand(batch_size,
                                                        self.num_writes,
                                                        self.memory_size).astype(np.float32)
            initial_precedence_weights /= initial_precedence_weights.sum(axis=2, keepdims=True) + 1
            # initial_linkage
            initial_linkage = memory_Addressing.TemporalLinkageState(
                link=tf.constant(initial_link),
                precedence_weights=tf.constant(initial_precedence_weights)
            )


            # initial_usage
            initial_usage = tf.zeros(shape=(batch_size, self.memory_size),dtype=dtype)

            # initial_access_state
            initial_access_state = AccessState(
                memory=tf.constant(initial_memory),
                read_weights=tf.constant(initial_read_weights),
                write_weights=tf.constant(initial_write_weights),
                linkage=initial_linkage,
                usage=initial_usage
            )
            return initial_access_state

    def initial_output(self, batch_size, dtype=tf.float32):
        return tf.zeros(shape=(batch_size, self.num_reads * self.word_size), dtype=dtype)
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

    # controller_input = tf.random_normal(
    #         shape=[batch_size,
    #         word_size*num_reads + 3*word_size + 5*num_reads + 3],
    #         dtype=tf.float64)
    #
    # input = memoryAccess.parse_input(controller_input)
    #
    # memory = tf.constant(np.random.rand(batch_size, memory_size, word_size))
    #
    # usage = tf.constant(np.random.rand(batch_size, memory_size))
    # write_weights = memoryAccess.write_weights(inputs=input,
    #                                            memory=memory,
    #                                            usage=usage)
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     write_weights = sess.run(write_weights)
    #     print(write_weights)

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
    # test Memory_Access._build

    controller_input = tf.constant(
        np.random.rand(batch_size,
                       word_size * num_reads + 3 * word_size + 5 * num_reads + 3).astype(np.float32))

    initial_access_state = memoryAccess.initial_state(batch_size=batch_size)
    print(initial_access_state)

    read_words, access_state = memoryAccess(inputs=controller_input,
                                            prev_access_state=initial_access_state)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("logs", sess.graph, filename_suffix='.memory_access')
        sess.run(tf.global_variables_initializer())
        read_words, access_state, initial_access_state = sess.run([read_words, access_state, initial_access_state])
        # print("read_words: \n", read_words.shape)
        # print("access_state: \n", access_state)
        print(initial_access_state)