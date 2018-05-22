# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import random
from model import Model
import utils

FLAGS = tf.flags.FLAGS

# Task parameters
tf.flags.DEFINE_integer("batch_size", 10,
                        "Batch size for training.")
tf.flags.DEFINE_integer("time", 12,
                        "the length for input sequence.")
tf.flags.DEFINE_integer("word_embedding_size", 3,
                        "the size for word embedding.")
tf.flags.DEFINE_integer("vocabulary_size", 547,
                        "the size for vocabulary")
tf.flags.DEFINE_string("train_file", "train.json",
                       "training file.")

# Training options.
tf.flags.DEFINE_integer("num_training_iterations", 1000,
                        "Number of iteration to train for.")
tf.flags.DEFINE_integer("report_interval", 100,
                        "Iterations between reports.")
tf.flags.DEFINE_string("checkpoint_dir", "./checkpoint",
                       "Checkpoint directory.")
tf.flags.DEFINE_integer("checkpoint_interval", 100,
                        "Checkpoint step interval.")

# Optimizer parameters
tf.flags.DEFINE_float("max_grad_norm", 50,
                      "Gradient clipping norm limit.")
tf.flags.DEFINE_float("learning_rate", 1e-4,
                      "Optimizer learning rate.")
tf.flags.DEFINE_float("optimizer_epsilon", 1e-10,
                      "Epsilon used for RMSProp optimizer.")


def train(vocabulary_size=FLAGS.vocabulary_size,
          embedding_size=FLAGS.word_embedding_size,
          num_training_iterations=FLAGS.num_training_iterations,
          report_interval=FLAGS.report_interval):
    # Define the placeholder
    inputs = tf.placeholder(
        shape=(None, None),
        dtype=tf.int64)
    labels = tf.placeholder(
        shape=[None], dtype=tf.int64)

    # Word embedding
    embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embed = tf.nn.embedding_lookup(embeddings, inputs)

    # Get model's logits
    # The shape for logits is [batch_size, max_classes]
    logits = Model().run_model(batch_size=FLAGS.batch_size,
                               input_sequence=embed)
    # Define loss function
    train_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)
    average_loss = tf.reduce_mean(train_loss, 0)

    # Set optimizer
    trainable_variables = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(train_loss, trainable_variables), FLAGS.max_grad_norm)

    global_step = tf.get_variable(
        name="global_step",
        shape=[],
        dtype=tf.int64,
        initializer=tf.zeros_initializer(),
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP]
    )

    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=FLAGS.learning_rate, epsilon=FLAGS.optimizer_epsilon
    )
    train_step = optimizer.apply_gradients(
        grads_and_vars=zip(grads, trainable_variables),
        global_step=global_step
    )

    # Define saver
    saver = tf.train.Saver()

    if FLAGS.checkpoint_interval > 0:
        hooks = [
            tf.train.CheckpointSaverHook(
                checkpoint_dir=FLAGS.checkpoint_dir,
                save_steps=FLAGS.checkpoint_interval,
                saver=saver
            )
        ]
    else:
        hooks = []

    # Train
    with tf.train.SingularMonitoredSession(
        hooks=hooks, checkpoint_dir=FLAGS.checkpoint_dir
    ) as sess:
        total_loss = 0
        # Read data set
        data_set = utils.encoding_and_read_data(raw_file=FLAGS.train_file)

        for train_iteration in range(num_training_iterations):
            # Read data.
            random.shuffle(data_set)

            # batch training
            batch_iteration = 0
            while 1:
                labels_batch, inputs_batch, index = utils.read_batch_data(
                    pos=batch_iteration*FLAGS.batch_size,
                    batch_size=FLAGS.batch_size,
                    data_set=data_set)
                if index != FLAGS.batch_size:
                    break
                # print(inputs_batch)
                labels_batch = np.squeeze(labels_batch)
                batch_iteration += 1

                # Run the session.
                _, average_loss_val = sess.run(fetches=[train_step, average_loss],
                                               feed_dict={inputs: inputs_batch,
                                                          labels: labels_batch})
                total_loss += average_loss_val

            # Report the result.
            if train_iteration % report_interval == 0:
                tf.logging.info("%d: Avg training loss %f.\n",
                                train_iteration, total_loss / (report_interval * batch_iteration))
                total_loss = 0


def main(unused_argv):
    tf.logging.set_verbosity(3)  # Print INFO log messages.
    train()


if __name__ == "__main__":
    tf.app.run()