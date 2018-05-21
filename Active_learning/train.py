# -*- coding: utf-8 -*-

import tensorflow as tf
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
        shape=(FLAGS.batch_size, FLAGS.time),
        dtype=tf.int64)
    labels = tf.placeholder(
        shape=FLAGS.batch_size, dtype=tf.int64)

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
        for train_iteration in range(num_training_iterations):
            # Read data.
            inputs_batch, labels_batch = utils.read_batch_data(
                    start_line=train_iteration*FLAGS.batch_size+1,
                    batch_size=FLAGS.batch_size)

            # Run the session.
            _, average_loss_val = sess.run(fetches=[train_step, average_loss],
                                           feed_dict={inputs: inputs_batch,
                                                      labels: labels_batch})
            total_loss += average_loss_val

            # Report the result.
            if train_iteration % report_interval == 0:
                tf.logging.info("%d: Avg training loss %f.\n",
                                train_iteration, total_loss / report_interval)
                total_loss = 0


def main(unused_argv):
    tf.logging.set_verbosity(3)  # Print INFO log messages.
    train()


if __name__ == "__main__":
    tf.app.run()