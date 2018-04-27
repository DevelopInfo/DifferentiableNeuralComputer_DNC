import tensorflow as tf




class Controller_Network():
    def __init__(self,
                num_units,
                name="controller_network",
                ):
        self.num_units = num_units
        self.name = name
        self.num_units = num_units

    def buildLSTM(self):
        self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.num_units)
        return self.lstm_cell

    def run_controller(self,
                       x_input_t,
                       r_vector_t_1,
                       initial_state
                       #self.lstm_cell.zero_state(x_input_t.shape[0], dtype=tf.float64)
                       ):
        input_t = tf.concat(values=[x_input_t, r_vector_t_1], axis=1)
        # expand dims
        input_t = tf.expand_dims(input=input_t, axis=1)

        output_t, state_t = tf.nn.dynamic_rnn(cell=self.lstm_cell,
                                              inputs=input_t,
                                              initial_state=initial_state)

        return output_t,state_t


# test
import numpy as np

if __name__ == "__main__":
    """ set parameters"""
    batch_size = 3
    V = 5
    W = 10
    R = 2

    x_input = np.ones(shape=(batch_size, 3, V))
    r_input = np.ones(shape=(batch_size, 3, W*R))
    print(x_input.shape[1])

    """build graph"""
    x_input_t = tf.placeholder(dtype=tf.float64, shape=(batch_size, V))
    r_input_t_1 = tf.placeholder(dtype=tf.float64, shape=(batch_size, W*R))

    controller = Controller_Network(num_units=2)
    initial_state = controller.buildLSTM().zero_state(batch_size=batch_size, dtype=tf.float64)
    output_t, state_t = controller.run_controller(x_input_t, r_input_t_1, initial_state)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(0, x_input.shape[1]):
            output_t_value, state_t_value = sess.run(
                [output_t, state_t],
                feed_dict={x_input_t:x_input[:,i,:], r_input_t_1:r_input[:,i,:]}
            )
            print(output_t_value)
            print(state_t_value)
            print(initial_state)






