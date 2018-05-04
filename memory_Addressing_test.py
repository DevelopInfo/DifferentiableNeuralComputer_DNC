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
        np.copyto(mem_data[0, 0], [1, 2])
        np.copyto(mem_data[0, 1], [3, 4])
        np.copyto(mem_data[0, 2], [5, 6])

        keys_data = np.random.randn(batch_size, num_heads, word_size)
        np.copyto(keys_data[0, 0], [5, 6])
        np.copyto(keys_data[0, 1], [1, 2])
        np.copyto(keys_data[0, 2], [5, 6])
        np.copyto(keys_data[0, 3], [3, 4])

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
        


if __name__ == '__main__':
    tf.test.main()