import gin
import tensorflow as tf
from evaluation.metrics import ConfusionMatrix
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from absl import flags
import numpy as np
from sklearn.metrics import f1_score

FLAGS = flags.FLAGS


class Evaluation:
    def __init__(self, model, ds_test, run_paths, num_classes):
        self.model = model
        self.ds_test = ds_test
        self.run_paths = run_paths
        self.num_classes = num_classes
        self.test_cm = ConfusionMatrix(self.num_classes)
        self._labels = []
        self._prediction = []
        self.save_path = os.path.join(self.run_paths["path_ckpts_eval"], 'confusionmatrix.png')

    # Evaluate the performance of the model with confusionmatrix, accuracy,
    # sensitivity, specificity and weighted f1-score
    def evaluate(self):
        checkpoint = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=self.model)
        if FLAGS.train:
            checkpoint.restore(tf.train.latest_checkpoint(self.run_paths['path_ckpts_train'])).expect_partial()
            logging.info(
                "===========================================start evaluate the last "
                "checkpoint.=======================================")
        else:
            logging.info(
                f"===========================================start evaluate the {FLAGS.index_ckpt} "
                f"checkpoint.=======================================")
            ckpt_file = "ckpt-" + str(FLAGS.index_ckpt)
            ckpt_path = os.path.join(self.run_paths['path_ckpts_train'], ckpt_file)
            checkpoint.restore(ckpt_path).expect_partial()

        for features, labels in self.ds_test:
            # test_loss, test_accuracy = model.evaluate(images, labels, verbose=1)
            prediction = self.model(features, training=False)
            prediction = tf.math.argmax(prediction, -1)
            labels = tf.squeeze(labels)

            self._labels = np.append(self._labels, labels.numpy())
            self._prediction = np.append(self._prediction, prediction.numpy())

            self.test_cm.update_state(labels, prediction, sample_weight=None)

        f1_s = f1_score(self._labels, self._prediction, average='weighted')
        sensitivity, specificity = self.test_cm.get_sensi_and_speci()
        accuracy = self.test_cm.get_accuracy()

        template = 'result: accuracy:{}, sensitivity: {}, specificity: {}, f1_score:{}, prediction:{}'
        logging.info(template.format(accuracy, sensitivity, specificity, f1_s, self._prediction))

        self.save_total_confusionmatrix()

    # Save the confusionmatrix to the save_path as confusionmatrix.png
    def save_total_confusionmatrix(self):
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
        logging.info(f"-------save a confusionmatrix.png to the path {self.run_paths['path_ckpts_eval']}------")
        logging.info(
            "===========================================finished"
            " evaluation.======================================================")

