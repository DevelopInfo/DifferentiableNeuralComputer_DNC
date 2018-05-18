# -*- coding: utf-8 -*-

import tensorflow as tf
from model import Model

FLAGS = tf.flags.FLAGS

# Task parameters
tf.flags.DEFINE_integer("batch_size", 10,
                        "Batch size for training.")
tf.flags.DEFINE_integer("time", 20,
                        "the length for input sequence.")
tf.flags.DEFINE_integer("word_embedding_size", 1,
                        "the size for word embedding.")

# Training options.
tf.flags.DEFINE_integer("num_training_iterations", 1000,
                        "Number of iteration to train for.")
tf.flags.DEFINE_integer("report_interval", 100,
                        "Iterations between reports.")
tf.flags.DEFINE_integer("checkpoint_dir", "./checkpoint",
                        "Checkpointing directory.")
tf.flags.DEFINE_integer("checkpoint_interval", 100,
                        "Checkpointing step interval.")

# Optimizer parameters
tf.flags.DEFINE_integer("max_grad_norm", 50,
                        "Gradient clipping norm limit.")
tf.flags.DEFINE_integer("learning_rate", 1e-4,
                        "Optimizer learning rate.")
tf.flags.DEFINE_integer("optimizer_epsilon", 1e-10,
                        "Epsilon used for RMSProp optimizer.")

def train(num_training_interations=FLAGS.num_training_interation,
          report_interval=FLAGS.report_interval):
    # Define the placeholder
    inputs = tf.placeholder(
        shape=(FLAGS.batch_size, FLAGS.time, FLAGS.word_embedding_size),
        dtype=tf.float32)
    labels = tf.placeholder(
        shape=FLAGS.batch_size, dtype=tf.int64)

    # Get model's logits
    # The shape for logits is [batch_size, max_classes]
    logits = Model().run_model(input_sequence=input)

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
        for train_iteration in range(num_training_interations):
            # Read data.

            # Run the session.
            _, average_loss_val = sess.run([train_step, average_loss])
            total_loss += average_loss_val

            # Report the result.
            if train_iteration % report_interval == 0:
                tf.logging.info("%d: Avg training loss %f.\n",
                                train_iteration, total_loss / report_interval)
                total_loss = 0

def main(unused_argv):
  tf.logging.set_verbosity(3)  # Print INFO log messages.
  train(FLAGS.num_training_iterations, FLAGS.report_interval)


if __name__ == "__main__":
  tf.app.run()