import math
import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    data = np.loadtxt("D:\Pycharm\PycharmProjects\\tensorflowModel\demoRecommendModel\dataSet\data.txt", dtype=float)
    # data_len = len(data)
    data_len = 6
    # 将样本存于第一个文件中
    filename = "D:\Pycharm\PycharmProjects\\tensorflowModel\demoRecommendModel\\tfrecords\saveTfrecordPath\sample_0.tfrecords"
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(0, math.floor(data_len/2)):
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(float_list=tf.train.FloatList(value=[data[i, 0]])),
            "feature_array": tf.train.Feature(float_list=tf.train.FloatList(value=data[i, 1:17]))
        }))
        writer.write(example.SerializeToString())
    writer.close()
    #将样本存于第二个文件中
    filename = "D:\Pycharm\PycharmProjects\\tensorflowModel\demoRecommendModel\\tfrecords\saveTfrecordPath\\sample_1.tfrecords"
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(math.floor(data_len/2), data_len):
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(float_list=tf.train.FloatList(value=[data[i, 0]])),
            "feature_array": tf.train.Feature(float_list=tf.train.FloatList(value=data[i, 1:17]))
        }))
        writer.write(example.SerializeToString())
    writer.close()