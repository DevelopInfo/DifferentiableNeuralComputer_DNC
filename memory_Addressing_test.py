"""Test the module memory_Addressing"""

import tensorflow as tf
import numpy as np
import memory_Addressing as addr

class CosineWeightsTest(tf.test.TestCase):

    def testShape(self):
        batch_size = 5
        num_heads = 3
        memory_size = 7
        word_size = 2

        module = addr.CosineWeights(num_heads=num_heads,
                                    word_size=word_size)
        memory = tf.placeholder(dtype=tf.float32,
                                shape=[batch_size, memory_size, word_size])
        keys = tf.placeholder(dtype=tf.float32,
                              shape=[batch_size, num_heads, word_size])
        strengths = tf.placeholder(dtype=tf.float32,
                                   shape=[batch_size, num_heads])
        cweights = module(memory=memory,
                          keys=keys,
                          strengths=strengths)
        self.assertTrue(cweights.get_shape().is_compatible_with(
            [batch_size, num_heads, memory_size]))

    def testValue(self):
        batch_size = 5
        num_heads = 4
        memory_size  = 7
        word_size = 2

        mem_data = np.random.randn(batch_size, memory_size, word_size)
        print("before: \n", mem_data)
        np.copyto(mem_data[0, 0], [1, 2])
        np.copyto(mem_data[0, 1], [3, 4])
        np.copyto(mem_data[0, 2], [5, 6])
        print("after: \n", mem_data)

        keys_data = np.random.randn(batch_size, num_heads, word_size)
        print("before: \n", keys_data)
        np.copyto(keys_data[0, 0], [5, 6])
        np.copyto(keys_data[0, 1], [1, 2])
        np.copyto(keys_data[0, 2], [5, 6])
        np.copyto(keys_data[0, 3], [3, 4])
        print("after: \n", keys_data)

        strengths_data = np.random.randn(batch_size, num_heads)

        module = addr.CosineWeights(num_heads=num_heads,
                                    word_size=word_size)
        memory = tf.placeholder(dtype=tf.float32,
                                shape=[batch_size, memory_size, word_size])
        keys = tf.placeholder(dtype=tf.float32,
                              shape=[batch_size, num_heads, word_size])
        strengths = tf.placeholder(dtype=tf.float32,
                                   shape=[batch_size, num_heads])
        cweights = module(memory=memory,
                          keys=keys,
                          strengths=strengths)

        with self.test_session() as sess:
            result = sess.run(
                cweights,
                feed_dict={memory: mem_data,
                           keys: keys_data,
                           strengths: strengths_data}
            )

        # Manually checks results
        strengths_softplus = np.log(1 + np.exp(strengths_data))
        similarity = np.zeros(shape=(memory_size))

        for b in range(batch_size):
            for h in range(num_heads):
                key = keys_data[b, h]
                key_norm = np.linalg.norm(key)

                for m in range(memory_size):
                    row = mem_data[b, m]
                    similarity[m] = np.dot(key, row) / (key_norm * np.linalg.norm(row))

                similarity = np.exp(similarity * strengths_softplus[b, h])
                similarity /= similarity.sum()
                self.assertAllClose(result[b, h], similarity, atol=1e-4, rtol=1e-4)

    def testDivideByZero(self):
        batch_size = 5
        num_heads = 4
        memory_size = 10
        word_size = 2

        module = addr.CosineWeights(num_heads, word_size)
        keys = tf.random_normal([batch_size, num_heads, word_size])
        strengths = tf.random_normal([batch_size, num_heads])

        # First row of memory is non-zero to concentrate attention on this location.
        # Remaining rows are all zero.
        first_row_ones = tf.ones([batch_size, 1, word_size], dtype=tf.float32)
        remaining_zeros = tf.zeros(
            [batch_size, memory_size - 1, word_size], dtype=tf.float32)
        mem = tf.concat((first_row_ones, remaining_zeros), 1)

        output = module(mem, keys, strengths)
        gradients = tf.gradients(output, [mem, keys, strengths])

        with self.test_session() as sess:
          output, gradients = sess.run([output, gradients])
          self.assertFalse(np.any(np.isnan(output)))
          self.assertFalse(np.any(np.isnan(gradients[0])))
          self.assertFalse(np.any(np.isnan(gradients[1])))
          self.assertFalse(np.any(np.isnan(gradients[2])))

if __name__ == '__main__':
    tf.test.main()