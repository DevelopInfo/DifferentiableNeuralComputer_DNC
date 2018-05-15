# -*- coding: utf-8 -*-


import tensorflow as tf

def read_and_decode(tfrecord_file, story_t, query_t, ans_t, ans_index_t):
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
    story = tf.reshape(story, (1, story_t))
    query = tf.decode_raw(features["query"], tf.int64)
    query = tf.reshape(query, (1, query_t))
    ans = tf.decode_raw(features["ans"], tf.int64)
    ans = tf.reshape(ans, (1, ans_t))
    ans_index = tf.decode_raw(features["ans_index"], tf.int64)
    ans_index = tf.reshape(ans_index, (1, ans_index_t))


    return story, query, ans, ans_index


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
            print(story_val.shape)