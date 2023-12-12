import gin
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
from input_pipeline.preprocessing import normalize, augment


@gin.configurable
def load(name, data_dir):
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")

        train_filename = data_dir + "/train.tfrecords"
        val_filename = data_dir + "/val.tfrecords"
        test_filename = data_dir + "/test.tfrecords"

        # get datasets from TFRecord
        raw_train_ds = tf.data.TFRecordDataset(train_filename)
        raw_val_ds = tf.data.TFRecordDataset(val_filename)
        raw_test_ds = tf.data.TFRecordDataset(test_filename)
        ds_info = "idrid"

        image_feature_description = {
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
        }

        def _parse_function(exam_proto):
            features = tf.io.parse_single_example(exam_proto, image_feature_description)
            img_raw = tf.io.decode_jpeg(features['image_raw'], channels=3)
            img_raw = tf.reshape(img_raw, [256, 256, 3])
            img_raw = tf.cast(img_raw, tf.float32)
            label = features['label']
            return (img_raw, label)

        ds_train = raw_train_ds.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = raw_val_ds.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = raw_test_ds.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        logging.info(f"finish Preparing dataset {name}...")
        
        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "eyepacs":
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'diabetic_retinopathy_detection/btgraham-300',
            split=['train', 'validation', 'test'],
            shuffle_files=True,
            with_info=True,
            data_dir=data_dir
        )
        logging.info(f"finish Preparing dataset {name}...")
        def _preprocess(img_label_dict):
            return img_label_dict['image'], img_label_dict['label']

        ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "mnist":
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train[:90%]', 'train[90%:]', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            data_dir=data_dir
        )
        logging.info(f"finish Preparing dataset {name}...")
        return prepare(ds_train, ds_val, ds_test, ds_info)

    else:
        raise ValueError

@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching):
    # Prepare train dataset
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.map(
        augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.shuffle(batch_size * 10)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info
