import math
import numpy as np
import random as rn
import tensorflow as tf

class FMOriginal(object):

    """
    初始化参数
    """
    def __init__(self, learning_rate, feature_length, embedding_length, batch_size, loss_type):
        self.learning_rate = learning_rate
        self.feature_length = feature_length
        self.embedding_length = embedding_length
        self.batch_size = batch_size
        self.loss_type = loss_type

    """
    输入与输出
    """
    def input_output(self):
        # 输入
        self.x = tf.placeholder(tf.float32, [None, self.feature_length], name="x_input")
        # 输出
        self.y_actual = tf.placeholder(tf.float32, [None, 1], name="y_actual")

    """
    前向传播
    """
    def forward_propagation(self):
        # 线性部分
        w0 = tf.get_variable(name="w0", shape=[1], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
        w = tf.get_variable(name="w", shape=[self.feature_length, 1], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
        self.linear = tf.add(tf.matmul(self.x, w), w0)   # None*1
        # 交互部分
        embedding = tf.get_variable(name="embedding", shape=[self.feature_length, self.embedding_length], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
        self.interaction = 0.5 * tf.reduce_sum(
            tf.subtract(
                tf.pow(
                    tf.matmul(self.x, embedding), 2),
                tf.matmul(tf.pow(self.x, 2), tf.pow(embedding, 2))
            ), axis=1, keepdims=True)
        self.predict_sum = tf.add(self.linear, self.interaction)  # None*1
        self.predict_sigmoid = tf.sigmoid(self.predict_sum)    # None*1

    """
    训练
    """
    def trains(self):
        if(self.loss_type is "crossEntropyLoss"):
            self.loss = tf.losses.log_loss(self.y_actual, self.predict_sigmoid)
        elif(self.loss_type is "logitLoss"):
            self.loss = -tf.log(tf.sigmoid(tf.multiply(self.y_actual, self.predict_sum)))
        else:
            print("损失函数类型错误，请重新输入！！！")
        self.mean_loss = tf.reduce_mean(self.loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        self.train = optimizer.minimize(self.mean_loss)

    """
    AUC
    """
    def model_auc(self):
        if (self.loss_type is "crossEntropyLoss"):
            self.auc_reset, self.auc_value = tf.metrics.auc(self.y_actual, self.predict_sigmoid)
        elif (self.loss_type is "logitLoss"):
            self.y_actual_conver = tf.maximum(self.y_actual, 0)
            self.predict_sum_conver = tf.sigmoid(self.predict_sum)
            self.auc_reset, self.auc_value = tf.metrics.auc(self.y_actual_conver, self.predict_sum_conver)
        else:
            print("损失函数类型错误，请重新输入！！！")

    """
    构建FMOriginal模型图
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
    #搭建FMOriginal网络
    model = FMOriginal(0.001, feature_length, 5, 100, "crossEntropyLoss")
    model.build_graph()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        batch_size = model.batch_size
        train_num = 200
        if(model.loss_type is "crossEntropyLoss"):
            for tn in range(1, train_num+1):
                sample_flag = rn.randint(0, train_data.shape[0] - batch_size)
                train_input = train_data[sample_flag: sample_flag+batch_size][:, 1:]
                train_label = train_data[sample_flag: (sample_flag+batch_size)][:, 0].reshape(batch_size, 1)
                feed_dict_train_sample = {model.x: train_input, model.y_actual: train_label}
                _, _, train_auc_value, train_mean_loss = sess.run([model.train, model.auc_reset, model.auc_value,
                                                                   model.mean_loss], feed_dict=feed_dict_train_sample)
                if (tn % 10 == 0):
                    # 训练样本
                    print("train_auc_value = ", train_auc_value, "train_mean_loss = ", train_mean_loss, end=" ")
                    #测试样本
                    feed_dict_test_data = {model.x: test_data[:, 1:],
                                            model.y_actual: test_data[:, 0].reshape(test_data.shape[0], 1)}
                    _, test_auc_value, test_mean_loss = sess.run([model.auc_reset, model.auc_value, model.mean_loss],
                                                                 feed_dict=feed_dict_test_data)
                    print("test_auc_value = ", test_auc_value, "test_mean_loss = ", test_mean_loss)
        elif(model.loss_type is "logitLoss"):
            for i in range(0, len(train_data)):
                if (train_data[i, 0] == 0):
                    train_data[i, 0] = -1
            for i in range(0, len(test_data)):
                if (test_data[i, 0] == 0):
                    test_data[i, 0] = -1
            for tn in range(1, train_num+1):
                sample_flag = rn.randint(0, train_data.shape[0] - batch_size)
                train_input = train_data[sample_flag: sample_flag+batch_size][:, 1:]
                train_label = train_data[sample_flag: (sample_flag+batch_size)][:, 0].reshape(batch_size, 1)
                feed_dict_train_sample = {model.x: train_input, model.y_actual: train_label}
                _, _, train_auc_value, train_mean_loss = sess.run([model.train, model.auc_reset, model.auc_value,
                                                                   model.mean_loss], feed_dict=feed_dict_train_sample)
                if (tn % 10 == 0):
                    # 训练样本
                    print("train_auc_value = ", train_auc_value, "train_mean_loss = ", train_mean_loss, end=" ")
                    #测试样本

                    feed_dict_test_data = {model.x: test_data[:, 1:],
                                            model.y_actual: test_data[:, 0].reshape(test_data.shape[0], 1)}
                    _, test_auc_value, test_mean_loss = sess.run([model.auc_reset, model.auc_value, model.mean_loss],
                                                                 feed_dict=feed_dict_test_data)
                    print("test_auc_value = ", test_auc_value, "test_mean_loss = ", test_mean_loss)
        else:
            print("损失函数类型错误，请重新输入！！！")