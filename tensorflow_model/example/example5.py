import tensorflow as tf

embedding_x = tf.Variable([[1.0, 2], [3, 4], [5, 6]], tf.float32)
sum_first = tf.reduce_sum(embedding_x, 0)
sum_first_square = tf.square(sum_first)

square_first = tf.square(embedding_x)
square_first_sum = tf.reduce_sum(square_first, 0)

inter = 0.5 * tf.reduce_sum(tf.subtract(sum_first_square, square_first_sum))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(sum_first))
    print(sess.run(sum_first_square))
    print(sess.run(square_first))
    print(sess.run(square_first_sum))
    print(sess.run(inter))
