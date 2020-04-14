import tensorflow as tf

sample = tf.Variable([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], tf.float32)  # 2 * 3
embedding = tf.Variable([[1, 1.1, 1.2, 1.3], [2, 2.1, 2.2, 3.3], [4, 4.1, 4.2, 4.3]], tf.float32)  # 3 * 4
sample_embedding = tf.Variable([[[1, 1.1, 1.2, 1.3], [2, 2.1, 2.2, 3.3], [4, 4.1, 4.2, 4.3]],
                                [[1, 1.1, 1.2, 1.3], [2, 2.1, 2.2, 3.3], [4, 4.1, 4.2, 4.3]]], tf.float32)  # 2 * 3 * 4
re_sample = tf.reshape(sample, [2, 3, -1])
y = tf.multiply(sample_embedding, re_sample)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print(sess.run(re_sample))
    print(sess.run(y))

