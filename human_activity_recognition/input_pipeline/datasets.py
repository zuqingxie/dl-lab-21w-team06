import gin
import logging
import tensorflow as tf
from input_pipeline.preprocessing import *
from absl import flags

@gin.configurable
def load(name, data_dir, window_length, window_shift):
    if name == "hapt":
        logging.info(f"Preparing dataset {name}...")
        # get datasets from TFRecord

        data_dir = data_dir + f'/wl{str(window_length)}_ws{str(window_shift)}'
        if flags.FLAGS.device_name == "local":
            train_filename = data_dir + "/train.tfrecords"
            val_filename = data_dir + "/val.tfrecords"
            test_filename = data_dir + "/test.tfrecords"

        elif flags.FLAGS.device_name == "local_debugging":
            train_filename = "C:\\Users\\xiezu\\Desktop\\train.tfrecords"
            val_filename = "C:\\Users\\xiezu\\Desktop\\val.tfrecords"
            test_filename = "C:\\Users\\xiezu\\Desktop\\test.tfrecords"

        elif flags.FLAGS.device_name == "GPU-Server":
            train_filename = data_dir + "/train.tfrecords"
            val_filename = data_dir + "/val.tfrecords"
            test_filename = data_dir + "/test.tfrecords"
        else:
            raise ValueError

        raw_train_ds = tf.data.TFRecordDataset(train_filename)
        raw_val_ds = tf.data.TFRecordDataset(val_filename)
        raw_test_ds = tf.data.TFRecordDataset(test_filename)



        feature_description = {
            'feature': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'label': tf.io.FixedLenFeature([], tf.string, default_value='')
        }

        def _parse_function(exam_proto):
            temp = tf.io.parse_single_example(exam_proto, feature_description)
            feature = tf.io.parse_tensor(temp['feature'], out_type=tf.float64)
            label = tf.io.parse_tensor(temp['label'], out_type=tf.int32)
            return (feature, label)

        ds_train = raw_train_ds.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = raw_val_ds.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = raw_test_ds.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)


        logging.info(f"finish preparing dataset {name}...")
        return prepare(ds_train, ds_val, ds_test)


@gin.configurable
def prepare(ds_train, ds_val, ds_test, batch_size, caching):
    # Prepare training dataset
    ds_train = ds_train.map(
       preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(batch_size * 10)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test
