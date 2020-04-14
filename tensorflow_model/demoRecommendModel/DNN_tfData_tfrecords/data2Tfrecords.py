import math
import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    data = np.loadtxt("..\data_set\data.txt", dtype=float)
    ## 训练样本
    train_data = data[0: 200]
    data_len = 200
    filename = "..\DNN_tfData_tfrecords\data_path\dataTrain.tfrecords"
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(0, data_len):
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(float_list=tf.train.FloatList(value=[data[i, 0]])),
            "feature_array": tf.train.Feature(float_list=tf.train.FloatList(value=data[i, 1:17]))
        }))
        writer.write(example.SerializeToString())
    writer.close()
    ## 测试样本
    test_data = data[200: 300]
    data_len = 100
    # 将样本存于第一个文件中
    filename = "..\DNN_tfData_tfrecords\data_path\dataTest.tfrecords"
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(0, data_len):
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(float_list=tf.train.FloatList(value=[data[i, 0]])),
            "feature_array": tf.train.Feature(float_list=tf.train.FloatList(value=data[i, 1:17]))
        }))
        writer.write(example.SerializeToString())
    writer.close()