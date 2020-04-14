import math
import numpy as np
import random as rn
import tensorflow as tf

class deepFMPaper(object):

    """
    初始化参数
    """
    def __init__(self, learning_rate, feature_length, embedding_length, batch_size, hidden_layer):
        self.learning_rate = learning_rate
        self.feature_length = feature_length
        self.embedding_length = embedding_length
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
        # batch feature embedding
        self.batch_fem = tf.placeholder(tf.float32, [None, self.feature_length, self.embedding_length], name="batch_fem")
        # keep_prob
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    """
    前向传播
    """
    def forward_propagation(self):
        """
        FM部分
        """
        # 线性部分
        w0 = tf.get_variable(name="w0", shape=[1], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
        w = tf.get_variable(name="w", shape=[self.feature_length, 1], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
        self.linear = tf.add(tf.matmul(self.x, w), w0)   # None*1
        # 交互部分
        self.embedding = tf.get_variable(name="embedding", shape=[self.feature_length, self.embedding_length],
                                         dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
        self.interaction = 0.5 * tf.reduce_sum(
            tf.subtract(
                tf.pow(
                    tf.matmul(self.x, self.embedding), 2),
                tf.matmul(tf.pow(self.x, 2), tf.pow(self.embedding, 2))
            ), axis=1, keepdims=True)
        self.predict_FM = tf.sigmoid(tf.add(self.linear, self.interaction))    # None*1

        """
        deep部分
        """
        embedding_extend = tf.multiply(self.batch_fem,
                                       tf.reshape(self.embedding,
                                                  [-1, self.feature_length, self.embedding_length]))  # None * n * k
        resample = tf.reshape(self.x, [-1, self.feature_length, 1])  # None * n * 1
        y_deep = tf.reshape(tf.multiply(embedding_extend, resample),
                            [-1, self.feature_length*self.embedding_length])  # None * (nk)
        for i in range(0, len(self.hidden_layer)):
            if i is 0:
                row = self.feature_length*self.embedding_length
            else:
                row = self.hidden_layer[i-1]
            wd = tf.Variable(tf.random_normal(shape=[row, self.hidden_layer[i]], mean=0, stddev=0.1))
            bias = tf.Variable(tf.random_normal(shape=[self.hidden_layer[i]], mean=0, stddev=0.01))
            y_deep = tf.nn.relu(tf.add(tf.matmul(y_deep, wd), bias))
            y_deep = tf.nn.dropout(y_deep, self.keep_prob)
        w_deep = tf.Variable(tf.random_normal(shape=[self.hidden_layer[len(self.hidden_layer)-1], 1], mean=0, stddev=0.1))
        bias_deep = tf.Variable(tf.random_normal(shape=[1], mean=0, stddev=0.01))
        self.predict_deep = tf.sigmoid(tf.add(tf.matmul(y_deep, w_deep), bias_deep))  # None * 1

        """
        FM和deep
        """
        wfd = tf.get_variable(name="wfd", shape=[1], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
        bfd = tf.get_variable(name="bfd", shape=[1], dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
        self.y_predict = tf.sigmoid(tf.add(wfd * tf.add(self.predict_FM, self.predict_deep), bfd))
        # self.y_predict = tf.sigmoid(tf.add(self.predict_FM, self.predict_deep))

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
    构建deepFMPaper模型图
    """
    def build_graph(self):
        self.input_output()
        self.forward_propagation()
        self.trains()
        self.model_auc()

if __name__ == "__main__":
    # 数据预处理#
    data = np.loadtxt("", dtype=float)
    train_data = data[0:math.ceil(len(data) * 0.6)]
    test_data = data[math.ceil(len(data) * 0.6):len(data)]
    feature_length = train_data.shape[1] - 1
    # 搭建deepFMPaper网络
    model = deepFMPaper(0.005, feature_length, 5, 100, [80, 80, 80])
    model.build_graph()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        batch_size = model.batch_size
        train_num = 1000
        for tn in range(1, train_num + 1):
            sample_flag = rn.randint(0, train_data.shape[0] - batch_size)
            train_input = train_data[sample_flag: sample_flag + batch_size][:, 1:]
            train_label = train_data[sample_flag: (sample_flag + batch_size)][:, 0].reshape(batch_size, 1)
            train_batch = np.ones([batch_size, model.feature_length, model.embedding_length])
            feed_dict_train_sample = {model.x: train_input, model.y_actual: train_label,
                                      model.batch_fem: train_batch, model.keep_prob: 0.5}
            _, _, train_auc_value, train_mean_loss = sess.run([model.train, model.auc_reset, model.auc_value,
                                                               model.mean_loss], feed_dict=feed_dict_train_sample)
            if (tn % 50 == 0):
                # 训练样本
                print("train_auc_value = ", train_auc_value, "train_mean_loss = ", train_mean_loss, end=" ")
                # 测试样本
                feed_dict_test_data = {model.x: test_data[:, 1:],
                                       model.y_actual: test_data[:, 0].reshape(test_data.shape[0], 1),
                                       model.batch_fem: np.ones([len(test_data), model.feature_length, model.embedding_length]),
                                       model.keep_prob: 1.0}
                _, test_auc_value, test_mean_loss = sess.run([model.auc_reset, model.auc_value, model.mean_loss],
                                                             feed_dict=feed_dict_test_data)
                print("test_auc_value = ", test_auc_value, "test_mean_loss = ", test_mean_loss)