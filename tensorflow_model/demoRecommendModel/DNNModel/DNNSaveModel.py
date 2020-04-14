import math
import numpy as np
import random as rn
import tensorflow as tf

class DNN(object):

    """
    初始化参数
    """
    def __init__(self, learning_rate, feature_length, batch_size, hidden_layer):
        self.learning_rate = learning_rate
        self.feature_length = feature_length
        self.batch_size = batch_size
        self.hidden_layer = hidden_layer

    """
    输入与输出
    """
    def input_output(self):
        # 输入
        self.x = tf.placeholder(tf.float32, [None, self.feature_length], name="x_input")
        # 输出
        self.y_actual = tf.placeholder(tf.float32, [None, 1], name="y_actual")
        # keep_prob
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    """
    前向传播
    """
    def forward_propagation(self):
        y_deep = self.x
        for i in range(0, len(self.hidden_layer)):
            if i is 0:
                row = self.feature_length
            else:
                row = self.hidden_layer[i-1]
            w = tf.Variable(tf.random_normal(shape=[row, self.hidden_layer[i]], mean=0, stddev=0.1))
            bias = tf.Variable(tf.random_normal(shape=[self.hidden_layer[i]], mean=0, stddev=0.01))
            y_deep = tf.nn.relu(tf.add(tf.matmul(y_deep, w), bias))
            y_deep = tf.nn.dropout(y_deep, self.keep_prob)
        w_output = tf.Variable(tf.random_normal(shape=[self.hidden_layer[len(self.hidden_layer)-1], 1], mean=0, stddev=0.1))
        bias_output = tf.Variable(tf.random_normal(shape=[1], mean=0, stddev=0.01))
        self.y_predict = tf.sigmoid(tf.add(tf.matmul(y_deep, w_output), bias_output), name="y_predict")


    """
    训练
    """
    def trains(self):
        self.loss = tf.losses.log_loss(self.y_actual, self.y_predict)
        self.mean_loss = tf.reduce_mean(self.loss, name="mean_loss")
        # self.mean_loss = -1.0 * tf.reduce_mean(
        #     tf.multiply(self.y_actual, tf.log(tf.clip_by_value(self.y_predict, 1e-10, 1.0))) +
        #     0.025 * tf.multiply((1.0 - self.y_actual), tf.log(1.0 - tf.clip_by_value(self.y_predict, 1e-10, 1.0)))
        # )
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        self.train = optimizer.minimize(self.mean_loss)

    """
    AUC
    """
    def model_auc(self):
        self.auc_reset, self.auc_value = tf.metrics.auc(self.y_actual, self.y_predict)

    """
    构建DNN模型图
    """
    def build_graph(self):
        self.input_output()
        self.forward_propagation()
        self.trains()
        self.model_auc()

if __name__ == "__main__":
    #数据预处理#
    data = np.loadtxt("D:\Pycharm\PycharmProjects\\tensorflowModel\demoRecommendModel\dataSet\data.txt", dtype=float)
    train_data = data[0:math.ceil(len(data)*0.6)]
    test_data = data[math.ceil(len(data)*0.6):len(data)]
    feature_length = train_data.shape[1] - 1
    #搭建DNN网络
    model = DNN(0.005, feature_length, 100, [16, 16, 16])
    model.build_graph()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        batch_size = model.batch_size
        train_num = 1000
        for tn in range(1, train_num+1):
            sample_flag = rn.randint(0, train_data.shape[0] - batch_size)
            train_input = train_data[sample_flag: sample_flag+batch_size][:, 1:]
            train_label = train_data[sample_flag: (sample_flag+batch_size)][:, 0].reshape(batch_size, 1)
            feed_dict_train_sample = {model.x: train_input, model.y_actual: train_label, model.keep_prob: 0.5}
            _, _, train_auc_value, train_mean_loss = sess.run([model.train, model.auc_reset, model.auc_value,
                                                               model.mean_loss], feed_dict=feed_dict_train_sample)
            if (tn % 50 == 0):
                # 训练样本
                print("train_auc_value = ", train_auc_value, "train_mean_loss = ", train_mean_loss, end=" ")
                #测试样本
                feed_dict_test_data = {model.x: test_data[:, 1:],
                                        model.y_actual: test_data[:, 0].reshape(test_data.shape[0], 1), model.keep_prob: 1.0}
                _, test_auc_value, test_mean_loss = sess.run([model.auc_reset, model.auc_value, model.mean_loss],
                                                             feed_dict=feed_dict_test_data)
                print("test_auc_value = ", test_auc_value, "test_mean_loss = ", test_mean_loss)
        saver = tf.train.Saver()
        saver.save(sess, "..\DNNModel\modelSavePath\DNNModel")