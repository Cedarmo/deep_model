import math
import numpy as np
import random as rn
import tensorflow as tf

def DNN_model(x, y_actual, feature_length, hidden_layer, keep_prob, learning_rate):
    y_deep = x
    for i in range(0, len(hidden_layer)):
        if i is 0:
            row = feature_length
        else:
            row = hidden_layer[i - 1]
        w = tf.Variable(tf.random_normal(shape=[row, hidden_layer[i]], mean=0, stddev=0.1))
        bias = tf.Variable(tf.random_normal(shape=[hidden_layer[i]], mean=0, stddev=0.01))
        y_deep = tf.nn.relu(tf.add(tf.matmul(y_deep, w), bias))
        y_deep = tf.nn.dropout(y_deep, keep_prob)
    w_output = tf.Variable(
        tf.random_normal(shape=[hidden_layer[len(hidden_layer) - 1], 1], mean=0, stddev=0.1))
    bias_output = tf.Variable(tf.random_normal(shape=[1], mean=0, stddev=0.01))
    y_predict = tf.sigmoid(tf.add(tf.matmul(y_deep, w_output), bias_output), name="y_predict")

    loss = tf.losses.log_loss(y_actual, y_predict)
    mean_loss = tf.reduce_mean(loss, name="mean_loss")
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    global_step = tf.train.get_or_create_global_step()
    train = optimizer.minimize(mean_loss, global_step=global_step)
    return y_predict, mean_loss, train, global_step


def main(_):
    # 数据预处理
    data = np.loadtxt("", dtype=float)
    train_data = data[0:math.ceil(len(data) * 0.6)]
    test_data = data[math.ceil(len(data) * 0.6):len(data)]
    batch_size = 100

    cluster = tf.train.ClusterSpec({"ps": FLAGS.strps_hosts.split(","), "worker": FLAGS.strwork_hosts.split(",")})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index,
                                                      cluster=cluster)):
            # 特征长度
            feature_length = 16
            x = tf.placeholder(tf.float32, [None, feature_length], name="x_input")
            y_actual = tf.placeholder(tf.float32, [None, 1], name="y_actual")
            keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            y_predict, mean_loss, train, global_step = DNN_model(x, y_actual, feature_length, [15, 10, 5], keep_prob,
                                                                 0.001)
            auc_reset, auc_value = tf.metrics.auc(y_actual, y_predict)

            hooks = [tf.train.StopAtStepHook(last_step=500)]
            checkpoint_dir = ""
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
                    feed_dict_train_sample = {x: train_input, y_actual: train_label, keep_prob: 0.5}
                    _, step, _, train_auc_value, train_mean_loss = sess.run(
                        [train, global_step, auc_reset, auc_value, mean_loss],
                        feed_dict=feed_dict_train_sample)
                    if (step % 50 == 0):
                        # 训练样本
                        print("step = ", step, "train_auc_value=", train_auc_value, "train_mean_loss=", train_mean_loss,
                              end=" ")
                        # 测试样本
                        feed_dict_test_data = {x: test_data[:, 1:],
                                               y_actual: test_data[:, 0].reshape(test_data.shape[0], 1),
                                               keep_prob: 1.0}
                        _, test_auc_value, test_mean_loss = sess.run([auc_reset, auc_value, mean_loss],
                                                                     feed_dict=feed_dict_test_data)
                        print("test_auc_value = ", test_auc_value, "test_mean_loss = ", test_mean_loss)


if __name__ == "__main__":
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string("strps_hosts", "localhost:2222,localhost:2223", "参数服务器")
    tf.app.flags.DEFINE_string("strwork_hosts", "localhost:2224,localhost:2225", "工作服务器")
    tf.app.flags.DEFINE_string("job_name", "ps", "参数服务器或者工作服务器")
    tf.app.flags.DEFINE_integer("task_index", 0, "job的task索引")
    tf.app.run()
