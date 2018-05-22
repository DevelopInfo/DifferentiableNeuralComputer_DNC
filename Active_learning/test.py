# -*- coding: utf-8 -*-

import tensorflow as tf
from model import Model
import utils
import numpy as np

FLAGS = tf.flags.FLAGS

# Task parameters
tf.flags.DEFINE_float("threshold", 0.07,
                      "the threshold value.")
tf.flags.DEFINE_integer("batch_size", 1,
                        "Batch size for testing.")
tf.flags.DEFINE_integer("word_embedding_size", 3,
                        "the size for word embedding.")
tf.flags.DEFINE_integer("vocabulary_size", 547,
                        "the size for vocabulary")
tf.flags.DEFINE_string("test_file", "test.json",
                       "test file.")

# Model checkpoint
tf.flags.DEFINE_string("checkpoint_dir", "./checkpoint/",
                       "Checkpoint directory.")


def test():
    # Define the placeholder.
    inputs = tf.placeholder(
        shape=(FLAGS.batch_size, None),
        dtype=tf.int64)

    # Word embedding
    embeddings = tf.Variable(
        tf.random_uniform([FLAGS.vocabulary_size, FLAGS.word_embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, inputs)

    # Run model.
    logits = Model().run_model(FLAGS.batch_size, embed)
    logits = tf.nn.softmax(logits)

    # Define saver
    saver = tf.train.Saver()

    # Reading checkpoints.
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)


    with tf.Session() as sess:
        # Loading model.
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print('No checkpoint file found.')

        # Read data set.
        data_set = utils.encoding_and_read_data(raw_file=FLAGS.test_file)

        # Read data and predict.
        batch_iteration = 0
        predict = []
        while 1:
            print(batch_iteration)
            # Read data.
            start_pos = batch_iteration*FLAGS.batch_size
            _, input_array, index = utils.read_batch_data(
                pos=start_pos,
                batch_size=FLAGS.batch_size,
                data_set=data_set)

            # End predict
            if index != FLAGS.batch_size:
                break

            # Predict
            logits_val = sess.run([logits], feed_dict={inputs: input_array})

            predict.append(logits_val[-1].max(axis=1))
            # for index, prob in enumerate(logits_val[-1].max(axis=1)):
            #     print(prob)
            #     if prob < FLAGS.threshold:
            #         input_retrain.append(input_array[index])

            # one iteration
            batch_iteration += 1
        predict_array = np.array(predict)
        predict_index = np.argsort(predict_array, axis=0)
        input_retrain = []
        for i in range(10):
            index = predict_index[i][0]
            input_retrain.append(data_set[index]["input"])
        # Write to train file
        utils.decode_and_write_data(input_retrain)


def main(unused_argv):
    tf.logging.set_verbosity(3)  # Print INFO log messages.
    test()


if __name__ == "__main__":
    tf.app.run()