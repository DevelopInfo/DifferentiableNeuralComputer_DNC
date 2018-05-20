import tensorflow as tf
import numpy as np

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("max_classes", 30, "The max number of classes")


class Model():
    """Test model"""
    def __init__(self, num_units=FLAGS.max_classes):
        self.num_units = num_units

    def run_model(self, batch_size, input_sequence, name="model"):
        """Run a model
        Args:
            input_sequence: shape = [batch_size, time, word_embedding_size],
            dtype = tf.float32

        Returns:
            logits: shape = [batch_size, time, max_classes]
            dtype = tf.float32
        """
        with tf.name_scope(name):
            lstm_cell = tf.contrib.rnn.LSTMCell(self.num_units)
            initial_state = lstm_cell.zero_state(
                batch_size=batch_size,
                dtype=tf.float32)
            output, _ = tf.nn.dynamic_rnn(
                cell=lstm_cell,
                inputs=input_sequence,
                initial_state=initial_state,
                dtype=tf.float32)
        return output[:, -1, :]


if __name__ == "__main__":
    """Set parameters"""
    batch_size = 3
    time = 2
    word_embedding_size = 1

    """Create some data"""
    input_sequence = np.random.randint(
        low=1, high=547,
        size=(batch_size, time, word_embedding_size)).astype(np.float32)
    model = Model()
    output = model.run_model(batch_size, input_sequence)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output_val = sess.run([output])
        print(output_val)
