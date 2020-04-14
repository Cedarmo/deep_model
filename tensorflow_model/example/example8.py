import tensorflow as tf

def parse_fn(example):
    example_fmt = {
             'features': tf.VarLenFeature(tf.int64),
             'label': tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(example, example_fmt)
    return parsed['features'], parsed['label']

files = tf.data.Dataset.list_files("")

data_set = files.apply(
    tf.contrib.data.parallel_interleave(lambda filename: tf.data.TFRecordDataset(filename),
                                        cycle_length=8))

data_set = data_set.map(map_func=parse_fn, num_parallel_calls=8)




data_set = data_set.batch(batch_size=32)
data_set = data_set.prefetch(buffer_size=32)

iterator = data_set.make_one_shot_iterator()
next_element = iterator.get_next()

