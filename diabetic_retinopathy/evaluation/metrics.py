import tensorflow as tf
import logging
import gin


# inherit tensorflow.keras.metrics.Metric as subclasses and overwrite methods
@gin.configurable
class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init__(self, f_betha, name="confusion_matrix", **kwargs):
        # run Parent constructor function
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.total_confusion_matrix = self.add_weight(name='total_confusion_matrix', shape=(2, 2), initializer="zeros")
        self.f_betha = f_betha

    def reset_states(self):
        for variables in self.variables:
            variables.assign(tf.zeros(shape=variables.shape))

    def update_state(self, y_true, y_pred):
        confusion_matrix = tf.math.confusion_matrix(tf.squeeze(y_true), y_pred, num_classes=2, dtype=tf.float32)
        self.total_confusion_matrix.assign_add(confusion_matrix)

    def get_evaluate_parameters(self):
        tn = self.total_confusion_matrix[0, 0]
        tp = self.total_confusion_matrix[1, 1]
        fp = self.total_confusion_matrix[0, 1]
        fn = self.total_confusion_matrix[1, 0]
        mini_num = tf.constant(1e-10)

        unbalanced_acc = (tp + tn) / (tp + fn + fp + tn + mini_num)
        sensitivity = tp / (tp + fn + mini_num)
        specificity = tn / (tn + fp + mini_num)
        balanced_acc = (sensitivity + specificity) / 2

        precision = tp / (tp + fp + mini_num)
        recall = tp / (tp + fn + mini_num)
        f1_score = 2 * (recall * precision) / (recall + precision + mini_num)

        F_betha_score = ((1 + self.f_betha ** 2) * tp) / (
                    (1 + self.f_betha ** 2) * tp + self.f_betha ** 2 * fn + fp)
        template = 'sensitivity: {}, specificity: {},' \
                   'balanced_acc: {}, unbalanced_acc: {}, ' \
                   'precision: {}, recall: {},F1_score: {}, F{}_score: {} '
        logging.info(template.format(sensitivity.numpy(), specificity.numpy(),
                                     balanced_acc.numpy(), unbalanced_acc.numpy(),
                                     precision.numpy(), recall.numpy(),
                                     f1_score.numpy(), self.f_betha, F_betha_score.numpy()))

    def result(self):
        return self.total_confusion_matrix


