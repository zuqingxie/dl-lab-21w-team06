import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2
from tensorflow.keras import backend as K


@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy>0, "float32") * tf.cast(x>0, "float32") * dy
    return tf.nn.relu(x), grad


class GuidedBackprop:
    def __init__(self, model, layerName=None):
        self.model = model
        self.layerName = layerName
        if self.layerName is None:
            self.layerName = self.find_target_layer()
        self.gb_model = self.get_guided_model()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply guided backpropagation")

    def get_guided_model(self):
        gb_model = Model(inputs = [self.model.inputs],
                         outputs = [self.model.get_layer(self.layerName).output])

        layer_dict = [layer for layer in gb_model.layers[1:] if hasattr(layer, "activation")]
        i = 1
        for layer in layer_dict:
            print(f"layer{i}: {layer.name}")
            if layer.activation == tf.nn.relu:
                layer.activation = guidedRelu
            i += 1

        return gb_model


    def guided_backpropagation(self, image):
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            tape.watch(inputs)
            outputs = self.gb_model(inputs)

        grads = tape.gradient(outputs, inputs)[0]
        grads = cv2.resize(np.asarray(grads), (3480,3480))
        grads = np.maximum(grads, 0)
        return grads

def deprocess_image(x):
    """Same normalization as in:
    https://github.com/rstudio/keras/blob/main/vignettes/examples/conv_filter_visualization.py
    """
    # normalize tensor: center on 0., ensure std is 0.25
    #x = (x + abs(x)) / 2
    x = x.copy()

    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x