# -*- coding: utf-8 -*-

import tensorflow as tf
from model import Model
import utils
import numpy as np

FLAGS = tf.flags.FLAGS

# Task parameters
tf.flags.DEFINE_float("threshold", 0.12,
                      "the threshold value.")
tf.flags.DEFINE_integer("batch_size", 4,
                        "Batch size for testing.")
tf.flags.DEFINE_integer("time", 12,
                        "the length for input sequence.")
tf.flags.DEFINE_integer("word_embedding_size", 3,
                        "the size for word embedding.")
tf.flags.DEFINE_integer("vocabulary_size", 547,
                        "the size for vocabulary")

# Model checkpoint
tf.flags.DEFINE_string("checkpoint_dir", "./checkpoint/",
                       "Checkpoint directory.")


def test():
    # Define the placeholder.
    inputs = tf.placeholder(
        shape=(None, FLAGS.time),
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

        # Read data.
        input_array, _ = utils.read_batch_data(2,FLAGS.batch_size)

        # Predict
        logits_val = sess.run([logits], feed_dict={inputs: input_array})

        input_retrain = []
        for index, prob in enumerate(logits_val[-1].max(axis=1)):
            if prob < FLAGS.threshold:
                input_retrain.append(input_array[index])
        input_retrain = np.array(input_retrain)
        utils.decode_and_write_data(input_retrain)


def main(unused_argv):
    tf.logging.set_verbosity(3)  # Print INFO log messages.
    test()


if __name__ == "__main__":
    tf.app.run()