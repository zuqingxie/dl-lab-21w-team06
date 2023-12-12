import tensorflow as tf


# inherit tf.keras.metrics.Metric.....as subclasses and overwrite its methods
class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init__(self, num_classes, name="confusion_matrix", **kwargs, ):
        # run Parent constructor function
        super(ConfusionMatrix, self).__init__(name=name, **kwargs, )
        self.num_classes = num_classes
        self.total_confusion_matrix = self.add_weight(name='total_confusion_matrix', shape=(num_classes, num_classes),
                                                      initializer="zeros")

    def reset_states(self):
        for variables in self.variables:
            variables.assign(tf.zeros(shape=variables.shape))

    def update_state(self, labels, prediction, sample_weight=None):
        confusion_matrix = tf.math.confusion_matrix(labels, prediction, dtype=tf.float32, num_classes=self.num_classes)
        self.total_confusion_matrix.assign_add(confusion_matrix)

    def result(self):
        return self.total_confusion_matrix

    def confusion_matrix(self, y_true, y_pred):
        confusion_matrix = tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32, num_classes=self.num_classes)
        return confusion_matrix

    # get sensiticity and specificity
    def get_sensi_and_speci(self):
        diag = tf.linalg.diag_part(self.total_confusion_matrix)
        sensitivity_specificity = diag / (tf.reduce_sum(self.total_confusion_matrix, 0) + tf.constant(1e-10))
        sensitivity = sensitivity_specificity.numpy()[1]
        specificity = sensitivity_specificity.numpy()[0]
        return sensitivity, specificity

    def get_accuracy(self):
        # diag sum of Joint probabilities
        accuracy = tf.linalg.trace(self.total_confusion_matrix) / (
                tf.reduce_sum(self.total_confusion_matrix) + tf.constant(1e-10))
        return accuracy * 100
