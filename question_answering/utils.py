# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import linecache

def read_and_decode(tfrecord_file, story_t, query_t, ans_t, ans_index_t):
    """Read the tfrecord file and decode the data"""
    filename_queue = tf.train.string_input_producer([tfrecord_file])
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized=serialized_example,
                                       features={
                                           "story": tf.FixedLenFeature([], tf.string),
                                           "query": tf.FixedLenFeature([], tf.string),
                                           "ans": tf.FixedLenFeature([], tf.string),
                                           "ans_index": tf.FixedLenFeature([], tf.string)
                                       })
    story = tf.decode_raw(features["story"], tf.int64)
    story = tf.reshape(story, (story_t, 1))
    story = tf.cast(story, tf.float32)
    query = tf.decode_raw(features["query"], tf.int64)
    query = tf.reshape(query, (query_t, 1))
    query = tf.cast(query, tf.float32)
    ans = tf.decode_raw(features["ans"], tf.int64)
    ans = tf.reshape(ans, (ans_t, 1))
    ans_index = tf.decode_raw(features["ans_index"], tf.int64)
    ans_index = tf.reshape(ans_index, (ans_index_t, 1))

    return story, query, ans, ans_index



def word_readable(lexicons_dict_file, word_index):
    """
    Args:
        lexicons_dict_file: the lexicons dictionary file
        word_index: a numpy.ndarray whose range is between 1 to 156
        and shape is `[batch_size, sequence_len]`
    Returns:
        return the word string which indicated by word_index
    """
    word = ""
    for i in range(word_index.shape[0]):
        word_line = ""
        for j in range(word_index.shape[1]):
            # print(word_index[i][j])
            if word_index[i][j] >= 1 or word_index[i][j] <= 156:
                word_line = word_line + linecache.getline(
                    filename=lexicons_dict_file,
                    lineno=word_index[i][j]).strip('\n') + " "
        word = word + "\n" + word_line
    word += "\n"
    return word

if __name__ == "__main__":
    story, query, ans, ans_index = read_and_decode(
        tfrecord_file="qa1_train.tfrecords",
        story_t=56,
        query_t=3,
        ans_t=1,
        ans_index_t=1
    )
    story_batch, query_batch, ans_batch = tf.train.shuffle_batch(
        tensors=[story, query, ans],
        batch_size=10,
        capacity=1000,
        min_after_dequeue=100
    )

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess=sess)
        for i in range(3):
            story_val, query_val = sess.run([story_batch, query_batch])
            word_readable(lexicons_dict_file="lexicons_dict.txt",
                          word_index=story_val.reshape(10,56).astype(np.int64))