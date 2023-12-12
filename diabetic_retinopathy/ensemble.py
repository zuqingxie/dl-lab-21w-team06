import gin
from input_pipeline import datasets
from sklearn.metrics import accuracy_score
import numpy as np
from utils import utils_params, utils_misc
import logging
from input_pipeline.create_TFRecord import create_tfrecords
from evaluation.metrics import ConfusionMatrix
from models.architectures import *
import math


@gin.configurable
def ensemble_voting(ds_test, models, voting_weights):
    test_cm = ConfusionMatrix()
    for test_images, test_labels in ds_test:
        total_pred = 0
        k = 0
        for model in models:
            predictions = model(test_images, training=False)
            predictions = np.array(predictions)
            for i in range(len(predictions)):
                a = predictions[i,0]
                b = predictions[i, 1]
                if a > 100:
                    a = 100
                elif a < -100:
                    a = -100

                if b > 100:
                    b = 100
                elif b < -100:
                    b = -100
                predictions[i,0] = math.exp(a) / (math.exp(a) + math.exp(b))
                predictions[i,1] = math.exp(b) / (math.exp(a) + math.exp(b))
            total_pred += predictions * voting_weights[k]
            k += 1

        ensembled_labels = np.argmax(total_pred, -1)
        test_labels = np.array(test_labels).flatten()
        _ = test_cm.update_state(test_labels, ensembled_labels)

    return accuracy_score(test_labels, ensembled_labels), test_cm

def ensemble():
    run_paths = utils_params.gen_run_folder()
    gin.parse_config_files_and_bindings(['../experiments_test/run1_2022-02-12T16-03-41-724633/ensemble1.gin'], [])
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO, b_stream=False)

    # setup pipeline
    if create_tfrecords():
        logging.info(
            f"Created TFRecords files at: {gin.query_parameter('create_tfrecords.record_dir')}")
    else:
        logging.info(
            "TFRecords files already exist. Proceed with the execution")
    ds_train, ds_val, ds_test, ds_info = datasets.load(
        data_dir=gin.query_parameter('create_tfrecords.record_dir'))


    model1 = CNN_team06()
    checkpoint1 = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=model1)
    path1 = tf.train.latest_checkpoint("../experiments_test/run1_2022-02-12T16-03-41-724633/ckpts")
    checkpoint1.restore(path1).expect_partial()
    model1.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics='SparseCategoricalAccuracy')
    gin.clear_config()

    gin.parse_config_files_and_bindings(['../experiments_test/run1_2022-02-15T06-51-32-270270/ensemble2.gin'], [])
    model2 = resnet_simple()
    checkpoint2 = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=model2)
    path2 = tf.train.latest_checkpoint('../experiments_test/run1_2022-02-15T06-51-32-270270/ckpts')
    checkpoint2.restore(path2).expect_partial()
    model2.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics='SparseCategoricalAccuracy')
    gin.clear_config()

    gin.parse_config_files_and_bindings(['../experiments_test/run1_2022-02-15T11-13-55-421071/ensemble3.gin'], [])
    model3 = vgg_like(input_shape=(256,256,3), n_classes=2)
    checkpoint3 = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=model3)
    path3 = tf.train.latest_checkpoint('../experiments_test/run1_2022-02-15T11-13-55-421071/ckpts')
    checkpoint3.restore(path3).expect_partial()
    model3.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics='SparseCategoricalAccuracy')

    models = [model1, model2, model3]

    acc, cm = ensemble_voting(ds_test, models)
    cm.get_evaluate_parameters()

    # Show accuracy
    template = 'accuracy{}'
    logging.info(template.format(acc))

    # Confusion matrix
    template = 'Confusion Matrix:\n{}'
    logging.info(template.format(cm.result().numpy()))

ensemble()
