import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    filename_list = tf.train.match_filenames_once("D:\Pycharm\PycharmProjects\\tensorflowModel\demoRecommendModel\\tfrecords\saveTfrecordPath\sample_*.tfrecords")
    filename_queue = tf.train.string_input_producer(filename_list, num_epochs=1, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialize_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialize_example,
                                   features={
                                       "label": tf.FixedLenFeature([], tf.float32),
                                       "feature_array": tf.FixedLenFeature([16], tf.float32)
                                   })
    label = features["label"]
    feature_array = features["feature_array"]
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(6):
            labelt, feature_arrayt = sess.run([label, feature_array])
            print(labelt, end=" ")
            print(feature_arrayt)
