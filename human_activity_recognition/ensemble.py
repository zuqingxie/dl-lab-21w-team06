import os
from evaluation.metrics import ConfusionMatrix
import tensorflow as tf
from models.lstm import *
import numpy as np
from input_pipeline import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import f1_score


@gin.configurable
class Ensemble:
    def __init__(self, num_classes, run_paths, weight_list):
        self.num_classes = num_classes
        self.run_paths = run_paths
        self.test_cm = ConfusionMatrix(self.num_classes)
        self._labels = []
        self._prediction = []
        self.weighted_prediction = [0] * 1740
        self.total_prediction = []
        self.path_model_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'experiments'))
        self.save_path = os.path.join(self.run_paths["path_ckpts_eval"], 'confusionmatrix.png')
        self.data_dir = "~/dl-lab-21w-team06/human_activity_recognition/input_pipeline"

        self.weight_list = weight_list

        # Setting the model type and it's parameters
        self.model_1 = rnn(window_size=300, lstm_units=112, dropout_rate=0.4, dense_units=64, rnn_name='lstm',
                           num_rnn=4,
                           activation='tanh')
        self.model_2 = rnn(window_size=300, lstm_units=128, dropout_rate=0.4, dense_units=32, rnn_name='gru', num_rnn=5,
                           activation='tanh')
        self.model_3 = rnn(window_size=300, lstm_units=112, dropout_rate=0.4, dense_units=64, rnn_name='gru', num_rnn=4,
                           activation='tanh')
        self.model_4 = rnn(window_size=300, lstm_units=96, dropout_rate=0.3, dense_units=32, rnn_name='gru', num_rnn=5,
                           activation='tanh')
        self.model_5 = rnn_mix(dense_units=32, dropout_rate=0.2, units=80, window_size=300)

        self.model = [self.model_1, self.model_2, self.model_3, self.model_4, self.model_5]

        # absolute path of all the models
        self.run_paths_ckpt_1 = "~/dl-lab-21w-team06/experiments_test/run2_2022-02-11T17-15-09-480006/ckpts"
        self.run_paths_ckpt_2 = "~/dl-lab-21w-team06/experiments_test/run2_2022-02-11T18-11-49-619451/ckpts"
        self.run_paths_ckpt_3 = "~/dl-lab-21w-team06/experiments_test/run2_2022-02-08T00-44-17-890571/ckpts"
        self.run_paths_ckpt_4 = "~/dl-lab-21w-team06/experiments_test/run2_2022-02-11T21-17-26-166146/ckpts"
        self.run_paths_ckpt_5 = "~/dl-lab-21w-team06/experiments_test/run2_2022-02-11T23-27-29-308747/ckpts"

        self.run_paths_ckpt = [self.run_paths_ckpt_1, self.run_paths_ckpt_2, self.run_paths_ckpt_3,
                               self.run_paths_ckpt_4, self.run_paths_ckpt_5]

    def average_evaluate(self):
        for i in range(len(self.model)):
            logging.info(f"starting evaluate the {i + 1}.th model")
            model_current = self.model[i]
            checkpoint = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=model_current)
            checkpoint.restore(tf.train.latest_checkpoint(self.run_paths_ckpt[i])).expect_partial()
            ds_train, ds_val, ds_test = datasets.load(name="hapt",
                                                      data_dir=self.data_dir,
                                                      window_length=300,
                                                      window_shift=100)

            for features, labels in ds_test:
                prediction = model_current(features, training=False)
                prediction = tf.math.argmax(prediction, -1)
                labels = tf.squeeze(labels)
                if i == 0:
                    self._labels = np.append(self._labels, labels.numpy())
                self._prediction = np.append(self._prediction, prediction.numpy())

            # sum them up with weight vector
            self.weighted_prediction = self.weighted_prediction + self.weight_list[i] * self._prediction
            self._prediction = []
            logging.info(f"finish evaluating the {i + 1}.th model")

        self.weighted_prediction = np.around(self.weighted_prediction)

        self.test_cm.update_state(self._labels, self.weighted_prediction, sample_weight=None)
        accuracy = self.test_cm.get_accuracy()
        sensitivity, specificity = self.test_cm.get_sensi_and_speci()

        template = 'result: accuracy:{}, sensitivity: {}, specificity: {}, prediction:{}'
        logging.info(template.format(accuracy, sensitivity, specificity, self._prediction))

        self.show_confusionmatrix()

    # save the confusionmatrix
    def show_confusionmatrix(self):
        plt.figure(figsize=(10, 10))
        confusion_matrix = self.test_cm.result().numpy()
        confusion_matrix_percentage = confusion_matrix / confusion_matrix.sum(axis=1)[:, None]
        confusion_matrix_percentage = np.around(confusion_matrix_percentage * 100)
        sns.heatmap(confusion_matrix_percentage, annot=True, fmt='.20g', cmap="YlGnBu")
        font_title = {'family': 'serif',
                      'color': 'black',
                      'weight': 'normal',
                      'size': 24,
                      }
        font_label = {'family': 'serif',
                      'color': 'darkblue',
                      'weight': 'normal',
                      'size': 18,
                      }
        plt.title('ConfusionMatrix', fontdict=font_title)
        plt.xlabel('Predicted labels', fontdict=font_label)
        plt.ylabel('True labels', fontdict=font_label)
        plt.savefig(self.save_path)
