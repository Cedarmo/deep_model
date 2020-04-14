import math
import numpy as np
import random as rn
import tensorflow as tf

def WAD_model(x, y_actual, batch_fem, feature_length, embedding_length, hidden_layer, keep_prob, learning_rate):
    """
    wide部分
    """
    wide = x  # None*n
    """
    deep部分
    """
    embedding = tf.get_variable(name="embedding", shape=[feature_length, embedding_length],
                                     dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
    embedding_extend = tf.multiply(batch_fem,
                                   tf.reshape(embedding,
                                              [-1, feature_length, embedding_length]))  # None * n * k
    resample = tf.reshape(x, [-1, feature_length, 1])  # None * n * 1
    y_deep = tf.reshape(tf.multiply(embedding_extend, resample),
                        [-1, feature_length * embedding_length])  # None * (nk)
    for i in range(0, len(hidden_layer)):
        if i is 0:
            row = feature_length * embedding_length
        else:
            row = hidden_layer[i - 1]
        wd = tf.Variable(tf.random_normal(shape=[row, hidden_layer[i]], mean=0, stddev=0.1))
        bias = tf.Variable(tf.random_normal(shape=[hidden_layer[i]], mean=0, stddev=0.01))
        y_deep = tf.nn.relu(tf.add(tf.matmul(y_deep, wd), bias))
        y_deep = tf.nn.dropout(y_deep, keep_prob)
    """
    wide和deep
    """
    wfd = tf.get_variable(name="wfd", shape=[feature_length + hidden_layer[len(hidden_layer) - 1], 1],
                          dtype=tf.float32,
                          initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1))
    bfd = tf.get_variable(name="bfd", shape=[1], dtype=tf.float32,
                          initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
    y_concat = tf.concat([wide, y_deep], 1)  # None * (n+隐藏层最后一层节点数量)
    y_predict = tf.sigmoid(tf.add(tf.matmul(y_concat, wfd), bfd), name="y_predict")
    """
    训练
    """
    loss = tf.losses.log_loss(y_actual, y_predict)
    mean_loss = tf.reduce_mean(loss, name="mean_loss")
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    global_step = tf.train.get_or_create_global_step()
    train = optimizer.minimize(mean_loss, global_step=global_step)
    return y_predict, mean_loss, train, global_step


def main(_):
    # 数据预处理
    data = np.loadtxt("D:\Pycharm\PycharmProjects\\tensorflowModel\demoRecommendModel\dataSet\data.txt", dtype=float)
    train_data = data[0:math.ceil(len(data) * 0.6)]
    test_data = data[math.ceil(len(data) * 0.6):len(data)]
    batch_size = 100

    cluster = tf.train.ClusterSpec({"ps": FLAGS.strps_hosts.split(","), "worker": FLAGS.strwork_hosts.split(",")})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):
            #特征长度
            feature_length = 16
            #隐向量长度
            embedding_length = 5
            # 输入
            x = tf.placeholder(tf.float32, [None, feature_length], name="x_input")
            # 输出
            y_actual = tf.placeholder(tf.float32, [None, 1], name="y_actual")
            # batch feature embedding
            batch_fem = tf.placeholder(tf.float32, [None, feature_length, embedding_length],
                                            name="batch_fem")
            # keep_prob
            keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            # 模型
            y_predict, mean_loss, train, global_step = WAD_model(x, y_actual, batch_fem, feature_length,
                                                                 embedding_length, [15, 10, 5], keep_prob, 0.001)
            auc_reset, auc_value = tf.metrics.auc(y_actual, y_predict)

            hooks = [tf.train.StopAtStepHook(last_step=500)]
            checkpoint_dir = "..\distributedModelSavePath\AsyBetWADModel"
            sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            with tf.train.MonitoredTrainingSession(server.target,
                                                   is_chief=(FLAGS.task_index == 0),
                                                   hooks=hooks,
                                                   checkpoint_dir=checkpoint_dir,
                                                   save_checkpoint_secs=2,
                                                   config=sess_config) as sess:
                while not sess.should_stop():
                    sample_flag = rn.randint(0, train_data.shape[0] - batch_size)
                    train_input = train_data[sample_flag: sample_flag + batch_size][:, 1:]
                    train_label = train_data[sample_flag: (sample_flag + batch_size)][:, 0].reshape(batch_size, 1)
                    train_batch = np.ones([batch_size, feature_length, embedding_length])
                    feed_dict_train_sample = {x: train_input, y_actual: train_label,
                                              batch_fem: train_batch, keep_prob: 0.5}
                    _, step, _, train_auc_value, train_mean_loss = sess.run(
                        [train, global_step, auc_reset, auc_value, mean_loss],
                        feed_dict=feed_dict_train_sample)
                    if (step % 50 == 0):
                        # 训练样本
                        print("step = ", step, "train_auc_value = ",
                              train_auc_value, "train_mean_loss = ", train_mean_loss, end=" ")
                        # 测试样本
                        feed_dict_test_data = {x: test_data[:, 1:],
                                               y_actual: test_data[:, 0].reshape(test_data.shape[0], 1),
                                               batch_fem: np.ones(
                                                   [len(test_data), feature_length, embedding_length]),
                                               keep_prob: 1.0}
                        _, test_auc_value, test_mean_loss = sess.run(
                            [auc_reset, auc_value, mean_loss],
                            feed_dict=feed_dict_test_data)
                        print("test_auc_value = ", test_auc_value, "test_mean_loss = ", test_mean_loss)

if __name__ == "__main__":
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string("strps_hosts", "localhost:2222,localhost:2223", "参数服务器")
    tf.app.flags.DEFINE_string("strwork_hosts", "localhost:2224,localhost:2225", "工作服务器")
    tf.app.flags.DEFINE_string("job_name", "worker", "参数服务器或者工作服务器")
    tf.app.flags.DEFINE_integer("task_index", 0, "job的task索引")
    tf.app.run()