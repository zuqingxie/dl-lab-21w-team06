import tensorflow as tf
import pandas as pd
import numpy as np
import gin
import os
from glob import glob


def normalize(df):
    data = df.copy()
    for column in data.columns:
        data[column] = (data[column] - data[column].mean()) / data[column].std()

    return data

def deleteEnd(df, wl, ws):
    a = len(df)
    end_length = (a - wl) % ws
    if end_length != 0:
        df = df.iloc[0:-end_length,:]
    return df

def deleteAllEnd(dic, wl, ws):
    for i in np.arange(0, len(dic), 1):
        dic[i] = deleteEnd(dic[i], wl=wl, ws=ws)
    return dic

def slice_dataframe(dic, wl, ws):
    idx = 0
    data = []
    label = []
    length = len(dic) - wl
    while idx <=  length:
        x = 0+idx
        y = wl+idx
        data_frac = dic.iloc[x:y, 0:6]
        # print(f'{idx} data_frac shape = {np.array(data_frac).shape}')
        data.append(np.array(data_frac))
        # a = dic.iloc[x:y, [6]].value_counts()
        # print(a)
        # print(f'a = {int(a.idxmax())}')
        a = getBigLabel(dic.iloc[x:y, [6]])
        label.append(int(a))

        idx += ws
    return data, label

def getBigLabel(df):
    a = df.groupby(['label']).size().reset_index(name='counts')
    b = a.sort_values('counts', ascending=False, ignore_index = True)
    c = b.at[0, 'label']
    return c

def appendList(feature, label, app_feature, app_label):
    for j in np.arange(0, len(app_feature), 1):

        feature.append(app_feature[j])
        label.append(int(app_label[j]))
    return feature, label

def creatDataFrame(dic, wl, ws):
    feature = []
    label =[]
    for i in np.arange(0, len(dic), 1):
        feature_tem, label_tem = slice_dataframe(dic[i], wl=wl, ws=ws)
        feature, label = appendList(feature, label, feature_tem, label_tem)
    c = {"features" : feature,
         "label" : label}
    df = pd.DataFrame(c)
    return df


def find_target_end(dic, idx, length, target):
    star = idx
    end = 0
    while idx < length:
        if dic.iat[idx, 6] != target:
            end = idx - 1
            break
        else:
            idx += 1

    if idx == length:
        end = length

    len = end - star + 1
    return idx, star, end, len

def get_target_fram(star, end, length, dic):
    if (star - length + 20) < 0:
        star = 0
    else:
        star = star - length + 20

    if (end + length - 20) > len(dic):
        end = len(dic)
    else:
        end = end + length - 20

    return dic.iloc[star:end,:]

def augmentTransition(dic, wl):
    idx = 0
    data = []
    label = []
    length = len(dic) - wl

    while idx < length:
        target = dic.iat[idx,6]
        if target > 6:
            idx, target_star, target_end, target_len = find_target_end(dic, idx, length, target)
            if target_len < (wl//3 + 20):
                continue
            elif (target_len >= (wl//3 +20)) & (target_len < (wl//2)):
                target_dic = get_target_fram(target_star, target_end, length = target_len, dic = dic)
                data_, label_ = slice_dataframe(target_dic, wl=wl, ws=1)
                data, label = appendList(data, label, data_, label_)
            elif target_len >= (wl//2):
                target_dic = get_target_fram(star = target_star, end = target_end, length = (wl//2), dic = dic)
                data_, label_ = slice_dataframe(target_dic, wl=wl, ws=1)
                data, label = appendList(data, label, data_, label_)
        else:
            idx += 1

    return data, label

def appendAugmentation(dic, raw_df, wl):
    aug_features = []
    aug_label = []
    for i in np.arange(0, len(dic), 1):
        feature, label = augmentTransition(dic[i], wl=wl)
        aug_features, aug_label = appendList(aug_features, aug_label, feature, label)

    c = {"features": aug_features,
         "label": aug_label}
    aug_df = pd.DataFrame(c)
    value = raw_df['label'].value_counts()
    max_num = value.max()
    df = raw_df.copy()
    for i in range(1, 13):
        label_num = value[i]
        diff = max_num - label_num
        if diff > 0:
            if i <= 6:
                df_tem = raw_df.loc[raw_df['label']==i]
                df_sampeled = df_tem.sample(n=diff, replace=True)
            else:
                df_tem = aug_df.loc[aug_df['label']==i]
                df_sampeled = df_tem.sample(n=diff, replace=True)

            df = pd.concat([df, df_sampeled], ignore_index=True)

    return df

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def to_example(ag_data, label):
    ag_data = tf.io.serialize_tensor(ag_data).numpy()
    label = tf.io.serialize_tensor(label).numpy()
    feature = {
        'feature': _bytes_feature(ag_data),
        'label': _bytes_feature(label)
    }
    example_proto =  tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


@gin.configurable
def create_tfrecords(window_length, window_shift, data_dir, records_dir):

    records_dir = records_dir + f'/wl{str(window_length)}_ws{str(window_shift)}'
    if os.path.exists(records_dir):
        return  False

    rawdata_path = data_dir
    rawlabel_path = data_dir + '/labels.txt'


    rawlabel_df = pd.read_table(rawlabel_path,
                                names=["experiment", "userID", "activity", "star_point", "end_point"],
                                delim_whitespace=True)
    acc_pattern = 'acc_exp*txt'
    gyro_pattern = 'gyro_exp*txt'
    acc_files = glob(os.path.join(rawdata_path, acc_pattern))
    gyro_files = glob(os.path.join(rawdata_path, gyro_pattern))
    acc_files.sort()
    gyro_files.sort()

    train_dic = {}
    val_dic = {}
    test_dic = {}

    data_df = pd.DataFrame()
    data_df_s = pd.DataFrame()
    experiment = -1


    for i in range(rawlabel_df.shape[0]):
        if rawlabel_df.at[i, 'experiment']-1 != experiment:
            experiment = rawlabel_df.at[i, 'experiment'] - 1

            path_acc = acc_files[experiment]
            acc_df = pd.read_table(path_acc, names=['a_x', 'a_y', 'a_z'], delim_whitespace=True)
            path_gyro = gyro_files[experiment]
            gyro_df = pd.read_table(path_gyro, names=['g_x', 'g_y', 'g_z'], delim_whitespace=True)
            data_df_s = pd.concat([acc_df, gyro_df], axis=1)
            data_df_s['label'] = 0

        label = rawlabel_df.at[i, 'activity']
        star = rawlabel_df.at[i, 'star_point'] - 1
        end = rawlabel_df.at[i, 'end_point']

        for j in np.arange(star, end, 1):
            data_df_s.iat[j, 6] = label


        if i != rawlabel_df.shape[0] - 1:
            if rawlabel_df.at[i+1, 'experiment']-1 != experiment:
                data_df_s = data_df_s.drop(data_df_s[data_df_s['label'] == 0].index)
                data_df_s = data_df_s.reset_index(drop=True)
                data_df_s.iloc[:, 0:6] = normalize(data_df_s.iloc[:, 0:6])
                # data_df = data_df.append(data_df_s, ignore_index=True)

                if experiment <= 42:
                    train_dic[experiment] = data_df_s
                elif (experiment >= 43) & (experiment <= 48):
                    val_dic[experiment-43] = data_df_s
                elif (experiment >= 49):
                    test_dic[experiment-49] = data_df_s

        elif i == rawlabel_df.shape[0] - 1:
            data_df_s = data_df_s.drop(data_df_s[data_df_s['label'] == 0].index)
            data_df_s = data_df_s.reset_index(drop=True)
            data_df_s.iloc[:, 0:6] = normalize(data_df_s.iloc[:, 0:6])
            # data_df = data_df.append(data_df_s, ignore_index=True)

            test_dic[experiment-49] = data_df_s

    train_dic = deleteAllEnd(train_dic, wl=window_length, ws=window_shift)
    val_dic = deleteAllEnd(val_dic, wl=window_length, ws=window_shift)
    test_dic = deleteAllEnd(test_dic, wl=window_length, ws=window_shift)


    train_df = creatDataFrame(train_dic, wl=window_length, ws=window_shift)
    aug_train_df = appendAugmentation(train_dic, train_df, wl=window_length)

    val_df = creatDataFrame(val_dic, wl=window_length, ws=window_shift)
    
    test_df = creatDataFrame(test_dic, wl=window_length, ws=window_shift)

    os.makedirs(records_dir)

    record_train = records_dir + "/train.tfrecords"
    with tf.io.TFRecordWriter(record_train) as writer:
        for (feature, label) in zip(aug_train_df['features'], aug_train_df['label']):
            train_example = to_example(feature, label)
            writer.write(train_example)


    record_val = records_dir + "/val.tfrecords"
    with tf.io.TFRecordWriter(record_val) as writer:
        for (feature, label) in zip(val_df['features'], val_df['label']):
            val_example = to_example(feature, label)
            writer.write(val_example)


    record_test = records_dir + "/test.tfrecords"
    with tf.io.TFRecordWriter(record_test) as writer:
        for (feature, label) in zip(test_df['features'], test_df['label']):
            test_example = to_example(feature, label)
            writer.write(test_example)

    return True
