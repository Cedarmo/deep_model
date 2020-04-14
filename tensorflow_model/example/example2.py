import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

with tf.variable_scope('V1', reuse=tf.AUTO_REUSE):
    a1 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))
    a2 = tf.Variable(tf.random_normal(shape=[2,3], mean=0, stddev=1), name='a2')
with tf.variable_scope('V2', reuse=tf.AUTO_REUSE):
    a3 = tf.get_variable(name='a1', shape=[1],initializer=tf.constant_initializer(1))
    a4 = tf.Variable(tf.random_normal(shape=[2,3], mean=0, stddev=1), name='a2')
    a5 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))
    a6 = tf.Variable(tf.random_normal(shape=[2, 3], mean=0, stddev=1), name='a2')
with tf.name_scope("V3"):
    a7 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))
    a8 = tf.Variable(tf.random_normal(shape=[2, 3], mean=0, stddev=1), name='a2')
    # a9 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))
    a10 = tf.Variable(tf.random_normal(shape=[2, 3], mean=0, stddev=1), name='a2')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(a1.name)
    print(a2.name)
    print(a3.name)
    print(a4.name)
    print(a5.name)
    print(a6.name)
    print(a7.name)
    print(a8.name)
    # print(a9.name)
    print(a10.name)


