import math
import numpy as np
import random as rn
import tensorflow as tf

class FMDropout(object):

    """
    初始化参数
    """
    def __init__(self, learning_rate, feature_length, embedding_length, batch_size):
        self.learning_rate = learning_rate
        self.feature_length = feature_length
        self.embedding_length = embedding_length
        self.batch_size = batch_size

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
        # 线性部分
        w = tf.get_variable(name="w", shape=[self.feature_length], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
        self.linear = tf.multiply(self.x, w)  # None*n
        self.linear_dropout = tf.nn.dropout(self.linear, self.keep_prob)   # None*n
        # 交互部分
        embedding = tf.get_variable(name="embedding", shape=[self.feature_length, self.embedding_length], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
        self.interaction = 0.5 * tf.subtract(
            tf.pow(
                tf.matmul(self.x, embedding), 2),
            tf.matmul(tf.pow(self.x, 2), tf.pow(embedding, 2)))     # None*k
        self.interaction_dropout = tf.nn.dropout(self.interaction, self.keep_prob)
        self.line_interaction_concat = tf.concat([self.linear_dropout, self.interaction_dropout], 1)  # None*(n+k)
        bias = tf.get_variable(name="bias", shape=[1], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
        wl = tf.get_variable(name="wl", shape=[self.feature_length + self.embedding_length, 1], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
        self.y_predict = tf.sigmoid(tf.add(tf.matmul(self.line_interaction_concat, wl), bias))    # None*1

    """
    训练
    """
    def trains(self):
        self.loss = tf.losses.log_loss(self.y_actual, self.y_predict)
        self.mean_loss = tf.reduce_mean(self.loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        self.train = optimizer.minimize(self.mean_loss)

    """
    AUC
    """
    def model_auc(self):
            self.auc_reset, self.auc_value = tf.metrics.auc(self.y_actual, self.y_predict)

    """
    构建FMDropout模型图
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
    #搭建FMDropout网络
    model = FMDropout(0.001, feature_length, 5, 100)
    model.build_graph()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        batch_size = model.batch_size
        train_num = 200
        for tn in range(1, train_num+1):
            sample_flag = rn.randint(0, train_data.shape[0] - batch_size)
            train_input = train_data[sample_flag: sample_flag+batch_size][:, 1:]
            train_label = train_data[sample_flag: (sample_flag+batch_size)][:, 0].reshape(batch_size, 1)
            feed_dict_train_sample = {model.x: train_input, model.y_actual: train_label, model.keep_prob: 0.5}
            _, _, train_auc_value, train_mean_loss = sess.run([model.train, model.auc_reset, model.auc_value,
                                                               model.mean_loss], feed_dict=feed_dict_train_sample)
            if (tn % 10 == 0):
                # 训练样本
                print("train_auc_value = ", train_auc_value, "train_mean_loss = ", train_mean_loss, end=" ")
                #测试样本
                feed_dict_test_data = {model.x: test_data[:, 1:],
                                        model.y_actual: test_data[:, 0].reshape(test_data.shape[0], 1), model.keep_prob: 1.0}
                _, test_auc_value, test_mean_loss = sess.run([model.auc_reset, model.auc_value, model.mean_loss],
                                                             feed_dict=feed_dict_test_data)
                print("test_auc_value = ", test_auc_value, "test_mean_loss = ", test_mean_loss)