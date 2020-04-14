import math
import numpy as np
import random as rn
import sklearn.metrics as skm
import tensorflow as tf

# 两个矩阵相乘
x = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
y = tf.reduce_mean(x, axis=1, keepdims=True)
z = tf.reduce_sum(x, 1)
u = tf.square(x)

with tf.Session() as sess:
    # print(sess.run(y))
    # print(sess.run(z))
    print(sess.run(u))
