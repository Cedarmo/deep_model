import math
import numpy as np
import random as rn
import tensorflow as tf

if __name__ == "__main__":
    #数据预处理#
    data = np.loadtxt("D:\Pycharm\PycharmProjects\\tensorflowModel\demoRecommendModel\dataSet\data.txt", dtype=float)
    test_data = data[math.ceil(len(data)*0.6):len(data)]
    actual_data = test_data[:, 0].reshape(test_data.shape[0], 1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.import_meta_graph("..\DNNModel\modelSavePath\DNNModel.meta")
        saver.restore(sess, "..\DNNModel\modelSavePath\DNNModel")

        graph = tf.get_default_graph()
        x_input = graph.get_operation_by_name("x_input").outputs[0]
        y_actual = graph.get_operation_by_name("y_actual").outputs[0]
        keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]
        feed_dict_test_data = {x_input: test_data[:, 1:],
                               y_actual: test_data[:, 0].reshape(test_data.shape[0], 1), keep_prob: 1.0}
        mean_loss, y_predict = sess.run([graph.get_operation_by_name("mean_loss").outputs[0],
                                           graph.get_operation_by_name("y_predict").outputs[0]],
                                            feed_dict=feed_dict_test_data)
        print(mean_loss)
        print(y_predict)
