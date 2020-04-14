import tensorflow as tf
import numpy as np

a = np.random.random([5, 3])
print(a)
y = tf.nn.embedding_lookup(a, [0, 6])
with tf.Session() as sess:
    print(sess.run(y))
    print(y)
