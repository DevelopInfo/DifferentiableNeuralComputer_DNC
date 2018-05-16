
import tensorflow as tf
import dnc_cell.dnc as dnc


FLAGS = tf.flags.FLAGS

# Model dnc parameters
tf.flags.DEFINE_integer("lstm_hidden_size", 64, "Size of LSTM hidden layer.")
tf.flags.DEFINE_integer("memory_size", 32, "The number of memory slots.")
tf.flags.DEFINE_integer("word_size", 56, "The width of each memory slot.")
tf.flags.DEFINE_integer("num_write_heads", 2, "Number of memory write heads.")
tf.flags.DEFINE_integer("num_read_heads", 4, "Number of memory read heads.")
tf.flags.DEFINE_integer("clip_value", 20,
                        "Maximum absolute value of controller and dnc outputs.")
tf.flags.DEFINE_integer("dnc_output_size", 1, "Size of dnc output.")

# Model fnn parameters
tf.flags.DEFINE_integer("fnn_hidden_size", 156, "Size of fnn hidden layer.")


class BabiModel():
    def __init__(self,
                 name="babi_model",
                 lstm_hidden_size=FLAGS.lstm_hidden_size,
                 memory_size=FLAGS.memory_size,
                 word_size=FLAGS.word_size,
                 num_write_heads=FLAGS.num_write_heads,
                 num_read_heads=FLAGS.num_read_heads,
                 clip_value=FLAGS.clip_value,
                 dnc_output_size = FLAGS.dnc_output_size,
                 fnn_hidden_size = FLAGS.fnn_hidden_size):
        self.name = name
        self.lstm_hidden_size = lstm_hidden_size
        self.memory_size = memory_size
        self.word_size = word_size
        self.num_write_heads = num_write_heads
        self.num_read_heads = num_read_heads
        self.clip_value = clip_value
        self.dnc_output_size = dnc_output_size
        self.fnn_hidden_size = fnn_hidden_size

    def __run_dnc(self, batch_size, input_sequence):
        access_config = {
            "memory_size": self.memory_size,
            "word_size": self.word_size,
            "num_reads": self.num_read_heads,
            "num_writes": self.num_write_heads
        }
        controller_config = {
            "num_units": self.lstm_hidden_size
        }
        clip_value = self.clip_value

        dnc_core = dnc.DNC(access_config=access_config,
                           controller_config=controller_config,
                           output_size=self.dnc_output_size,
                           clip_value=clip_value)
        initial_state = dnc_core.initial_state(batch_size=batch_size)
        output_sequence, _ = tf.nn.dynamic_rnn(cell=dnc_core,
                                               inputs=input_sequence,
                                               initial_state=initial_state)
        return output_sequence

    def __run_fnn(self, batch_size, input_sequence):
        """Run fully connection neural network
        Args:
            batch_size: the number of input sequence
            input_sequence: a tensor whose shape is `[batch_size, input_size]`
        Return:
            return a tensor whose shape is `[batch_size, output_size]`
        """
        with tf.name_scope("fnn"):
            weight = tf.Variable(
                initial_value=tf.random_normal(
                    shape=(input_sequence.get_shape().as_list()[-1], self.fnn_hidden_size)))
            bias = tf.Variable(
                initial_value=tf.random_normal(
                    shape=(batch_size, self.fnn_hidden_size)))
            output_sequence = tf.matmul(input_sequence, weight)
            output_sequence += bias
            output_sequence = tf.nn.relu(output_sequence)
            # output_sequence = tf.nn.softmax(logits=output_sequence)
            return output_sequence

    def run_model(self, batch_size, story_input_sequence, query_input_sequence):
        """Runs model on input sequence.
        Args:
            batch_size: the number of input sequence
            story_input_sequence: a tensor whose shape is `[batch_size, sequence_len, vec_size]`
            query_input_sequence: a tensor whose shape is `[batch_size, sequence_len, vec_sie]`
        Returns:
            output_sequence: a tensor whose shape is `[batch_size, one_hot_size]`
        """
        # run dnc
        with tf.variable_scope("story_dnc"):
            story_output_sequence = self.__run_dnc(batch_size, story_input_sequence)
        with tf.variable_scope("query_dnc"):
            query_output_sequence = self.__run_dnc(batch_size, query_input_sequence)

        # flatten the sequence
        flatten = tf.layers.Flatten()
        story_output_sequence = flatten(story_output_sequence)
        query_output_sequence = flatten(query_output_sequence)

        # concat
        fnn_input_sequence = tf.concat(values=[story_output_sequence, query_output_sequence],
                                       axis=-1)

        # run fnn
        output_sequence = self.__run_fnn(batch_size, fnn_input_sequence)
        #print(output_sequence)
        return output_sequence

import numpy as np

if __name__ == "__main__":
    batch_size = 10
    story_input_sequence = np.random.randint(size=(batch_size, 12, 1),
                                             low=1, high=156).astype(np.float32)
    query_input_sequence = np.random.randint(size=(batch_size, 12, 1),
                                             low=1, high=156).astype(np.float32)

    babi_model = BabiModel()
    output_sequence = babi_model.run_model(batch_size,
                                           story_input_sequence,
                                           query_input_sequence)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output_sequence = sess.run(output_sequence)


