import tensorflow as tf
import collections

"""Content-based addressing"""


class CosineWeights():
    """Cosine-weighted attention.

    Calculates the cosine similarity between a query and each word in memory, then
    applies a weighted softmax to return a sharp distribution.
    """

    def __init__(self,
               num_heads,
               word_size,
               name='cosine_weights'):
        """Initializes the CosineWeights module.

        Args:
          num_heads: number of memory heads.
          word_size: memory word size.
          name: module name (default 'cosine_weights')
        """
        self.name=name
        self.num_heads = num_heads
        self.word_size = word_size

    def __call__(self, memory, keys, strengths):
        return self.build(memory, keys, strengths)

    def build(self, memory, keys, strengths):
        """Connects the CosineWeights module into the graph.

        Args:
          memory: A 3-D tensor of shape `[batch_size, memory_size, word_size]`.
          keys: A 3-D tensor of shape `[batch_size, num_heads, word_size]`.
          strengths: A 2-D tensor of shape `[batch_size, num_heads]`.

        Returns:
          Weights tensor of shape `[batch_size, num_heads, memory_size]`.
        """
        # Calculates the inner product between the query vector and words in memory.
        # dot.shape = [batch_size, num_heads, memory_size]
        dot = tf.matmul(keys, memory, adjoint_b=True)

        # Outer product to compute denominator (euclidean norm of query and memory).
        # memory_norms.shape = [batch_size, memory_size, 1]
        memory_norms = self.vector_norms(memory)
        # key_norms.shape = [bat_size, num_heads, 1]
        key_norms = self.vector_norms(keys)
        # norm.shape = [batch, num_heads, memory_size]
        norm = tf.matmul(key_norms, memory_norms, adjoint_b=True)

        # Calculates cosine similarity between the query vector and words in memory.
        # similarity.shape = [batch_size, num_heads, memory_size]
        similarity = dot / norm

        return self.weighted_softmax(similarity, strengths)

    def vector_norms(self, m):
        squared_norms = tf.reduce_sum(m * m, axis=2, keepdims=True)
        return tf.sqrt(squared_norms)

    def weighted_softmax(self, activations, strengths):
        """Returns softmax over activations multiplied by positive strengths.

        Args:
          activations: A tensor of shape `[batch_size, num_heads, memory_size]`, of
            activations to be transformed. Softmax is taken over the last dimension.
          strengths: A tensor of shape `[batch_size, num_heads]` containing strengths to
            multiply by the activations prior to the softmax.

        Returns:
          A tensor of same shape as `activations` with weighted softmax applied.
        """
        # transformed_strengths.shape = [batch_size, num_heads, 1]
        transformed_strengths = tf.expand_dims(strengths, -1)
        # sharp_activations.shape = [batch_size, num_heads, memory_size]
        sharp_activations = activations * transformed_strengths
        # softmax_weights.shape = [batch_size, num_heads, memory_size]
        softmax_weights = tf.nn.softmax(sharp_activations, axis=2)
        return softmax_weights


"""Dynamic memory allocation"""


class Allocation():
    """Memory usage that is increased by writing and decreased by reading.

    This module is a pseudo-RNNCore whose state is a tensor with values in
    the range [0, 1] indicating the usage of each of `memory_size` memory slots.

    The usage is:

    *   Increased by writing, where usage is increased towards 1 at the write
      addresses.
    *   Decreased by reading, where usage is decreased after reading from a
      location when free_gate is close to 1.

    The function `write_allocation_weights` can be invoked to get free locations
    to write to for a number of write heads.
    """
    def __init__(self,
                 memory_size,
                 name='allocation'):
        self.name = name
        self.memory_size = memory_size

    def write_allocation_weights(self, usage, write_gates, num_writes):
        """Calculates freeness-based locations for writing to.

        This finds unused memory by ranking the memory locations by usage, for each
        write head. (For more than one write head, we use a "simulated new usage"
        which takes into account the fact that the previous write head will increase
        the usage in that area of the memory.)

        Args:
          usage: A tensor of shape `[batch_size, memory_size]` representing
              current memory usage.
          write_gates: A tensor of shape `[batch_size, num_writes]` with values in
              the range [0, 1] indicating how much each write head does writing
              based on the address returned here (and hence how much usage
              increases).
          num_writes: The number of write heads to calculate write weights for.

        Returns:
          tensor of shape `[batch_size, num_writes, memory_size]` containing the
              freeness-based write locations. Note that this isn't scaled by
              `write_gate`; this scaling must be applied externally.
        """
        with tf.name_scope('write_allocation_weights'):
            # expand gatings over memory locations
            write_gates = tf.expand_dims(write_gates, -1)

            allocation_weights = []
            for i in range(num_writes):
                allocation_weights.append(self.get_allocation(usage))
                # update usage to take into account writing to this new allocation
                usage += ((1 - usage) * write_gates[:, i, :] * allocation_weights[i])

            # Pack the allocation weights for the write heads into one tensor.
            return tf.stack(allocation_weights, axis=1)


    def get_allocation(self, usage):
        r"""Computes allocation by sorting `usage`.

        This corresponds to the value a = a_t[\phi_t[j]] in the paper.

        Args:
          usage: tensor of shape `[batch_size, memory_size]` indicating current
              memory usage. This is equal to u_t in the paper when we only have one
              write head, but for multiple write heads, one should update the usage
              while iterating through the write heads to take into account the
              allocation returned by this function.

        Returns:
          Tensor of shape `[batch_size, memory_size]` corresponding to allocation.
        """
        with tf.name_scope('allocation'):
            nonusage = 1 - usage
            #
            # sorted_usage.shape = [batch_size, memory_size]
            # indices.shape = [batch_size, memory_size]
            sorted_nonusage, indices = tf.nn.top_k(
                nonusage, k=self.memory_size, name='sort')
            # sorted_usage.shape = [batch_size, memory_size]
            sorted_usage = 1 - sorted_nonusage
            # prod_sorted_usage.shape = [batch_size, memory_size]
            prod_sorted_usage = tf.cumprod(sorted_usage, axis=1, exclusive=True)
            # sorted_allocation.shape = [batch_size, memory_size]
            sorted_allocation = sorted_nonusage * prod_sorted_usage

            inverse_indices = self.batch_invert_permutation(indices)

            # This final line "unsorts" sorted_allocation, so that the indexing
            # corresponds to the original indexing of `usage`.
            # allocation.shape = [batch_size, memory_size]
            allocation = self.batch_gather(sorted_allocation, inverse_indices)
            # return sorted_allocation, indices, inverse_indices, allocation
            return allocation

    def batch_invert_permutation(self, permutations):
        """Returns batched `tf.invert_permutation` for every row in `permutations`."""
        with tf.name_scope('batch_invert_permutation', values=[permutations]):
            unpacked = tf.unstack(permutations)
            inverses = [tf.invert_permutation(permutation) for permutation in unpacked]
            return tf.stack(inverses)

    def batch_gather(self, values, indices):
        """Returns batched `tf.gather` for every row in the input."""
        with tf.name_scope('batch_gather', values=[values, indices]):
            unpacked = zip(tf.unstack(values), tf.unstack(indices))
            result = [tf.gather(value, index) for value, index in unpacked]
            return tf.stack(result)

    def get_usage(self, write_weights, free_gates, read_weights, prev_usage):
        """Calculates the new memory usage u_t.

        Memory that was written to in the previous time step will have its usage
        increased; memory that was read from and the controller says can be "freed"
        will have its usage decreased.

        Args:
          write_weights: tensor of shape `[batch_size, num_writes,
              memory_size]` giving write weights at previous time step.
          free_gates: tensor of shape `[batch_size, num_reads]` which indicates
              which read heads read memory that can now be freed.
          read_weights: tensor of shape `[batch_size, num_reads,
              memory_size]` giving read weights at previous time step.
          prev_usage: tensor of shape `[batch_size, memory_size]` giving
              usage u_{t - 1} at the previous time step, with entries in range
              [0, 1].

        Returns:
          tensor of shape `[batch_size, memory_size]` representing updated memory
          usage.
        """
        # Calculation of usage is not differentiable with respect to write weights.
        write_weight = tf.stop_gradient(write_weights)
        usage = self.usage_after_write(prev_usage, write_weight)
        usage = self.usage_after_read(usage, free_gates, read_weights)
        return usage

    def usage_after_write(self, prev_usage, write_weights):
        """Calcualtes the new usage after writing to memory.

        Args:
          prev_usage: tensor of shape `[batch_size, memory_size]`.
          write_weights: tensor of shape `[batch_size, num_writes, memory_size]`.

        Returns:
          New usage, a tensor of shape `[batch_size, memory_size]`.
        """
        with tf.name_scope('usage_after_write'):
            # Calculate the aggregated effect of all write heads
            # write_weights.shape = [batch_size, memory_size]
            write_weights = 1 - tf.reduce_prod(1 - write_weights, [1], keepdims=False)
            # usage_after_write.shape = [batch_size, memory_size]
            usage_after_write = prev_usage + (1 - prev_usage) * write_weights
            return usage_after_write

    def usage_after_read(self, prev_usage, free_gates, read_weights):
        """Calcualtes the new usage after reading and freeing from memory.

        Args:
          prev_usage: tensor of shape `[batch_size, memory_size]`.
          free_gates: tensor of shape `[batch_size, num_reads]` with entries in the
              range [0, 1] indicating the amount that locations read from can be
              freed.
          read_weights: tensor of shape `[batch_size, num_reads, memory_size]`.

        Returns:
          New usage, a tensor of shape `[batch_size, memory_size]`.
        """
        with tf.name_scope('usage_after_read'):
            # free_gate.shape = [batch_size, num_reads, 1]
            free_gate = tf.expand_dims(free_gates, -1)
            # free_read_weights.shape = [batch_size, num_reads, memory_size]
            free_read_weights = free_gate * read_weights
            # phi.shape = [batch_size, memory_size]
            phi = tf.reduce_prod(1 - free_read_weights, [1], name='phi')
            return prev_usage * phi


"""Temporal memory linkage"""
TemporalLinkageState = collections.namedtuple('TemporalLinkageState',
                                              ('link', 'precedence_weights'))


class TemporalLinkage():
    """Keeps track of write order for forward and backward addressing.

    This is a pseudo-RNNCore module, whose state is a pair `(link,
    precedence_weights)`, where `link` is a (collection of) graphs for (possibly
    multiple) write heads (represented by a tensor with values in the range
    [0, 1]), and `precedence_weights` records the "previous write locations" used
    to build the link graphs.

    The function `directional_read_weights` computes addresses following the
    forward and backward directions in the link graphs.
    """

    def __init__(self, memory_size, num_writes, name='temporal_linkage'):
        """Construct a TemporalLinkage module.

        Args:
          memory_size: The number of memory slots.
          num_writes: The number of write heads.
          name: Name of the module.
        """
        self.memory_size = memory_size
        self.num_writes = num_writes
        self.name = name

    def build(self, write_weights, prev_state):
        """Calculate the updated linkage state given the write weights.

        Args:
          write_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
              containing the memory addresses of the different write heads.
          prev_state: `TemporalLinkageState` tuple containg a tensor `link` of
              shape `[batch_size, num_writes, memory_size, memory_size]`, and a
              tensor `precedence_weights` of shape `[batch_size, num_writes,
              memory_size]` containing the aggregated history of recent writes.

        Returns:
          A `TemporalLinkageState` tuple `next_state`, which contains the updated
          link and precedence weights.
        """
        link = self.link(prev_state.link, prev_state.precedence_weights,
                          write_weights)
        precedence_weights = self.precedence_weights(prev_state.precedence_weights,
                                                      write_weights)
        return TemporalLinkageState(
            link=link, precedence_weights=precedence_weights)

    def directional_read_weights(self, link, prev_read_weights, forward):
        """Calculates the forward or the backward read weights.

        For each read head (at a given address), there are `num_writes` link graphs
        to follow. Thus this function computes a read address for each of the
        `num_reads * num_writes` pairs of read and write heads.

        Args:
          link: tensor of shape `[batch_size, num_writes, memory_size,
              memory_size]` representing the link graphs L_t.
          prev_read_weights: tensor of shape `[batch_size, num_reads,
              memory_size]` containing the previous read weights w_{t-1}^r.
          forward: Boolean indicating whether to follow the "future" direction in
              the link graph (True) or the "past" direction (False).

        Returns:
          tensor of shape `[batch_size, num_reads, num_writes, memory_size]`
        """
        with tf.name_scope('directional_read_weights'):
            # We calculate the forward and backward directions for each pair of
            # read and write heads; hence we need to tile the read weights and do a
            # sort of "outer product" to get this.
            expanded_read_weights = tf.stack([prev_read_weights] * self.num_writes,
                                             1)
            result = tf.matmul(expanded_read_weights, link, adjoint_b=forward)
            # Swap dimensions 1, 2 so order is [batch, reads, writes, memory]:
            return tf.transpose(result, perm=[0, 2, 1, 3])

    def precedence_weights(self, prev_precedence_weights, write_weights):
        """Calculates the new precedence weights given the current write weights.

        The precedence weights are the "aggregated write weights" for each write
        head, where write weights with sum close to zero will leave the precedence
        weights unchanged, but with sum close to one will replace the precedence
        weights.

        Args:
          prev_precedence_weights: A tensor of shape `[batch_size, num_writes,
              memory_size]` containing the previous precedence weights.
          write_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
              containing the new write weights.

        Returns:
          A tensor of shape `[batch_size, num_writes, memory_size]` containing the
          new precedence weights.
        """
        # print(prev_precedence_weights)
        # print(write_weights)
        with tf.name_scope('precedence_weights'):
            write_sum = tf.reduce_sum(write_weights, 2, keep_dims=True)
            return (1 - write_sum) * prev_precedence_weights + write_weights

    def link(self, prev_link, prev_precedence_weights, write_weights):
        """Calculates the new link graphs.

        For each write head, the link is a directed graph (represented by a matrix
        with entries in range [0, 1]) whose vertices are the memory locations, and
        an edge indicates temporal ordering of writes.

        Args:
          prev_link: A tensor of shape `[batch_size, num_writes, memory_size,
              memory_size]` representing the previous link graphs for each write
              head.
          prev_precedence_weights: A tensor of shape `[batch_size, num_writes,
              memory_size]` which is the previous "aggregated" write weights for
              each write head.
          write_weights: A tensor of shape `[batch_size, num_writes, memory_size]`
              containing the new locations in memory written to.

        Returns:
          A tensor of shape `[batch_size, num_writes, memory_size, memory_size]`
          containing the new link graphs for each write head.
        """
        # print(prev_link)
        # print(prev_precedence_weights)
        # print(write_weights)
        with tf.name_scope('link'):
          batch_size = prev_link.get_shape()[0].value
          write_weights_i = tf.expand_dims(write_weights, 3)
          write_weights_j = tf.expand_dims(write_weights, 2)
          prev_precedence_weights_j = tf.expand_dims(prev_precedence_weights, 2)
          prev_link_scale = 1 - write_weights_i - write_weights_j
          new_link = write_weights_i * prev_precedence_weights_j
          link = prev_link_scale * prev_link + new_link
          # Return the link with the diagonal set to zero, to remove self-looping
          # edges.
          return tf.matrix_set_diag(
              link,
              tf.zeros(
                  [batch_size, self.num_writes, self.memory_size],
                  dtype=link.dtype))

# test

import numpy as np

if __name__ == "__main__":
    """set the parameters"""
    batch_size = 3
    num_writes = 2
    num_reads = 1
    memory_size = 4
    word_size = 5

    ##############################################
    ##############################################
    # test class CosineWeights
    # cosineWeights = CosineWeights(num_heads=num_writes, word_size=word_size)

    ###############################################
    # test CosineWeights.build
    # memory = tf.constant(np.random.rand(batch_size, memory_size, word_size))
    # keys = tf.constant(np.random.rand(batch_size, num_writes, word_size))
    # strengths = tf.constant(np.random.rand(batch_size, num_writes) + np.ones(shape=[batch_size, num_writes]))
    # consine_Weights = cosineWeights.build(memory=memory,
    #                     keys=keys,
    #                     strengths=strengths)
    # with tf.Session() as sess:
    #     consine_Weights = sess.run(consine_Weights)
    #     print(consine_Weights)

    ##############################################
    ##############################################
    # test class Allocation
    allocation = Allocation(memory_size=memory_size)

    ################################################
    # test Allocation.get_usage

    # write_weights = tf.constant(np.random.rand(batch_size, num_writes, memory_size))
    # free_gates = tf.constant(np.random.rand(batch_size, num_reads))
    # read_weights = tf.constant(np.random.rand(batch_size, num_reads, memory_size))
    # prev_usage = tf.constant(np.random.rand(batch_size, memory_size))
    # usage = allocation.get_usage(write_weights=write_weights,
    #                      free_gates=free_gates,
    #                      read_weights=read_weights,
    #                      prev_usage=prev_usage)
    # with tf.Session() as sess:
    #     usage = sess.run(usage)
    #     print(usage)

    ##############################################
    # test Allocation.get_allocation

    # usage = tf.constant(np.random.rand(batch_size, memory_size))
    # sorted_allocation, indices, inverse_indices, allocation = allocation.get_allocation(usage=usage)
    #
    # with tf.Session() as sess:
    #     sorted_allocation, indices, inverse_indices, allocation = sess.run([sorted_allocation, indices, inverse_indices, allocation])
    #     print(sorted_allocation)
    #     print(indices)
    #     print(inverse_indices)
    #     print(allocation)

    ################################################
    # test Allocation.write_allocation_weights

    # usage = tf.constant(np.random.rand(batch_size, memory_size))
    # write_gates = tf.constant(np.random.rand(batch_size, num_writes))
    # allocation_weights = allocation.write_allocation_weights(
    #     usage=usage,
    #     write_gates=write_gates,
    #     num_writes=num_writes
    # )
    #
    # with tf.Session() as sess:
    #     allocation_weights = sess.run(allocation_weights)
    #     print(allocation_weights)

    ##############################################
    ##############################################
    # test class TemporalLinkage

    # temporalLinkage = TemporalLinkage(memory_size=memory_size, num_writes=num_writes)

    #############################################
    # test TemporalLinkage.precedence_weights

    # write_weights = tf.constant(np.random.rand(batch_size, num_writes, memory_size ))
    # prev_precedence_weights = tf.constant(np.random.rand(batch_size, num_writes, memory_size))
    #
    # precedence_weights = temporalLinkage.precedence_weights(
    #     prev_precedence_weights=prev_precedence_weights,
    #     write_weights=write_weights)
    #
    # with tf.Session() as sess:
    #     precedence_weights = sess.run([precedence_weights])
    #     print(precedence_weights)

    ###########################################################
    # test TemporalLinkage.link

    # prev_link = tf.constant(np.random.rand(batch_size, num_writes, memory_size, memory_size))
    # write_weights = tf.constant(np.random.rand(batch_size, num_writes, memory_size))
    # prev_precedence_weights = tf.constant(np.random.rand(batch_size, num_writes, memory_size))
    #
    # link = temporalLinkage.link(prev_link=prev_link,
    #                             prev_precedence_weights=prev_precedence_weights,
    #                             write_weights=write_weights)
    #
    # with tf.Session() as sess:
    #     link = sess.run([link])
    #     print(link)

    #########################################################
    # test TemporalLinkage.build

    # prev_TemporalLinkageState = collections.namedtuple('prev_TemporalLinkageState',
    #                                               ('link', 'precedence_weights'))
    # write_weights = tf.constant(np.random.rand(batch_size, num_writes, memory_size))
    # prev_TemporalLinkageState.link = tf.constant(np.random.rand(batch_size, num_writes, memory_size, memory_size))
    # prev_TemporalLinkageState.precedence_weights = tf.constant(np.random.rand(batch_size, num_writes, memory_size))
    #
    # temporalLinkageState = temporalLinkage.build(write_weights=write_weights,
    #                                              prev_state=prev_TemporalLinkageState)
    #
    # with tf.Session() as sess:
    #     temporalLinkageState = sess.run(temporalLinkageState)
    #     print(temporalLinkageState)

