import tensorflow as tf
from evaluation.metrics import ConfusionMatrix
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from visualization import visualize


class Evaluation:

    def __init__(self, model, ds_test, run_paths):

        self.test_cm = ConfusionMatrix()
        self.model = model
        self.ds_test = ds_test
        self.run_paths = run_paths
        self.save_path = os.path.join(self.run_paths["path_ckpts_eval"], 'confusionmatrix.png')

    def evaluate(self):
        logging.info(
            "===========================================start evaluate the last "
            "checkpoint.=======================================")
        checkpoint = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(), model=self.model)
        checkpoint.restore(tf.train.latest_checkpoint(self.run_paths['path_ckpts_train'])).expect_partial()
        # self.model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['SparseCategoricalAccuracy'])

        for images, labels in self.ds_test:
            # self.model.evaluate(images, labels, verbose=1)
            prediction = self.model(images, training=False)
            prediction = tf.math.argmax(prediction, -1)
            self.test_cm.update_state(labels, prediction)
        self.test_cm.get_evaluate_parameters()
        self.save_total_confusionmatrix()

        visualize(self.model, layername=None, save_path=self.run_paths["path_ckpts_eval"])

    def save_total_confusionmatrix(self):
        plt.figure(figsize=(4, 4))
        sns.heatmap(self.test_cm.result(), annot=True, cmap=plt.cm.Blues)
        plt.title('ConfusionMatrix')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.savefig(self.save_path)
        logging.info(f"-------save a confusionmatrix.png to the path {self.run_paths['path_ckpts_eval']}------")
        logging.info(
            "===========================================finished"
            " evaluation.======================================================")
