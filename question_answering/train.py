# -*- coding: utf-8 -*-

import tensorflow as tf
import question_answering.utils as utils
import question_answering.model_v1 as model

FLAGS = tf.flags.FLAGS

# Optimizer parameters.
tf.flags.DEFINE_float("max_grad_norm", 50, "Gradient clipping norm limit.")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Optimizer learning rate.")
tf.flags.DEFINE_float("optimizer_epsilon", 1e-10,
                      "Epsilon used for RMSProp optimizer.")

# Task parameters
tf.flags.DEFINE_integer("batch_size", 10, "Batch size for training.")
tf.flags.DEFINE_integer("story_len", 56, "the length of story.")
tf.flags.DEFINE_integer("query_len", 3, "the length of query.")
tf.flags.DEFINE_integer("one_hot_size", 156, "the size of one_hot vector")

# Training options.
tf.flags.DEFINE_integer("num_training_iterations", 100000,
                        "Number of iterations to train for.")
tf.flags.DEFINE_integer("report_interval", 100,
                        "Iterations between reports (samples, valid loss).")
tf.flags.DEFINE_string("checkpoint_dir", "./tmp",
                       "Checkpointing directory.")
tf.flags.DEFINE_integer("checkpoint_interval", -1,
                        "Checkpointing step interval.")

def train(num_training_interations, report_interval):
    story, query, ans, ans_index = utils.read_and_decode(
        tfrecord_file="qa1_train.tfrecords",
        story_t=FLAGS.story_len,
        query_t=FLAGS.query_len,
        ans_t=1,
        ans_index_t=1
    )

    story_batch, query_batch, ans_batch = tf.train.shuffle_batch(
        tensors=[story, query, ans],
        batch_size=FLAGS.batch_size,
        capacity=1000,
        min_after_dequeue=100
    )

    # Define the input of model and the ans
    story_input = tf.placeholder(shape=(FLAGS.batch_size, FLAGS.story_len, 1),
                                 dtype=tf.float32,
                                 name="story_input")
    query_input = tf.placeholder(shape=(FLAGS.batch_size, FLAGS.query_len, 1),
                                 dtype=tf.float32,
                                 name="query_input")
    answer = tf.placeholder(shape=FLAGS.batch_size,
                            dtype=tf.int64,
                            name="answer")

    babiMode = model.BabiModel()
    logits = babiMode.run_model(batch_size=FLAGS.batch_size,
                                story_input_sequence=story_batch,
                                query_input_sequence=query_batch)
    # Define the loss function
    train_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.squeeze(ans_batch)
    )
    average_loss = tf.reduce_mean(input_tensor=train_loss, axis=0)

    # Define the accuracy
    top_1_op = tf.nn.in_top_k(logits, tf.squeeze(ans_batch), 1)
    correct_prediction = tf.equal(top_1_op,True)
    average_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Set up optimizer with global norm clipping.
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
            hooks=hooks, checkpoint_dir=FLAGS.checkpoint_dir) as sess:
        writer = tf.summary.FileWriter("logs", sess.graph)
        tf.train.start_queue_runners(sess=sess)
        total_loss = 0
        total_accuracy = 0

        for train_iteration in range(num_training_interations):
            _, average_loss_val, average_accuracy_val = sess.run(
                fetches=[train_step, average_loss, average_accuracy])
            total_loss += average_loss_val
            total_accuracy += average_accuracy_val
            if train_iteration % report_interval == 0:
                tf.logging.info("%d: Avg training loss %f.\n",
                                train_iteration, total_loss / report_interval)
                total_loss = 0
                tf.logging.info("%d: Avg training accuracy %f.\n",
                                train_iteration, total_accuracy / report_interval)
                total_accuracy = 0

def main(unused_argv):
  tf.logging.set_verbosity(3)  # Print INFO log messages.
  train(FLAGS.num_training_iterations, FLAGS.report_interval)


if __name__ == "__main__":
  tf.app.run()