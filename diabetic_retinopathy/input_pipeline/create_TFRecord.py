import os.path
import gin
import tensorflow as tf
import pathlib
import pandas as pd
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from input_pipeline.preprocessing import preprocess

# Convert to a 2-class classification
def label2two(df):
    for i in df.index:
        if df.loc[i,'Retinopathy grade'] <= 1:
            df.loc[i,'Retinopathy grade'] = 0
        else:
            df.loc[i,'Retinopathy grade'] = 1
    return df


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(image_string, label):
    feature = {
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def resample(df_label):
    for i in range(len(df_label)):
        if  df_label.loc[i, 'Retinopathy grade'] == 1:
            df_label.loc[i, "weights"] = 0
        else:
            df_label.loc[i, "weights"] = 1
        i += 1
    counter = df_label["Retinopathy grade"].value_counts()
    a = counter[1]
    b = counter[0]
    if a >= b:
        diff = a - b
        df_label = df_label.append(
            df_label.sample(n=diff, replace=True, weights=df_label["weights"]).reset_index(drop=True))
    else:
        diff = b - a
        df_label = df_label.append(
            df_label.sample(n=diff, replace=True, weights=df_label["Retinopathy grade"]).reset_index(drop=True))

    df_label = df_label.groupby("Retinopathy grade")
    print(df_label)
    df_data = df_label.sample(frac=1)
    df_data = shuffle(df_data)
    return df_data

def write_tfrecord(record_name, df):
    record_file = record_name
    with tf.io.TFRecordWriter(record_file) as writer:
        for (filename, label) in zip(df["address"], df["Retinopathy grade"]):
            image = tf.io.read_file(filename)
            image, label = preprocess(image, label, 256, 256)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            success, image_string = cv2.imencode(".jpeg", image)
            img_bytes = image_string.tobytes()
            tf_example = image_example(img_bytes, int(label))
            writer.write(tf_example.SerializeToString())

@gin.configurable
def create_tfrecords(data_dir, record_dir):
    train_record_dir = record_dir + "/train.tfrecords"
    val_record_dir = record_dir + "/val.tfrecords"
    test_record_dir = record_dir + "/test.tfrecords"
    if os.path.exists(train_record_dir) & os.path.exists(val_record_dir) & os.path.exists(test_record_dir):
        return False

    train_image_dir = data_dir + '/images/train'
    test_image_dir = data_dir + '/images/test'
    train_val_label_dir = data_dir + '/labels/train.csv'
    test_label_dir = data_dir + '/labels/test.csv'
    train_val_label = pd.read_csv(train_val_label_dir)
    test_label = pd.read_csv(test_label_dir)

    train_val_label = train_val_label.dropna(axis=1)
    test_label = test_label.dropna(axis=1)

    train_val_label = label2two(train_val_label)
    test_label = label2two(test_label)

    #add train_val_data directory
    train_image_paths = list(pathlib.Path(train_image_dir).glob('*.jpg'))
    train_image_paths = [str(path) for path in train_image_paths]
    train_val_label['address'] = train_image_paths

    #add test_data directory
    test_image_paths = list(pathlib.Path(test_image_dir).glob('*.jpg'))
    test_image_paths = [str(path) for path in test_image_paths]
    test_label['address'] = test_image_paths

    # split train dataset into train dataset and validation dataset
    train_label, val_data = train_test_split(train_val_label, test_size=0.2)
    train_label.reset_index(drop=True, inplace=True)
    val_data.reset_index(drop=True, inplace=True)
    #resample train dataset
    train_data = resample(train_label)


    #create tfrecords
    write_tfrecord(record_name=train_record_dir, df=train_data)
    write_tfrecord(record_name=val_record_dir, df=val_data)
    write_tfrecord(record_name=test_record_dir, df=test_label)

    return True