from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName

        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM")

    def compute_heatmap(self, image):
        gradModel = Model(
            inputs = [self.model.inputs],
            outputs = [self.model.get_layer(self.layerName).output, self.model.output]
        )

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]

        grads = tape.gradient(loss, convOutputs)

        convOutputs = convOutputs[0]
        guideGrads = grads[0]

        weights = tf.reduce_mean(guideGrads, axis=(0, 1)) #global average pooling
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        Relu_cam = np.maximum(cam, 0)


        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(Relu_cam, (w, h))

        # normalize the heatmap
        heatmap = Relu_cam / heatmap.max()
        heatmap = (heatmap * 255).astype("uint8")

        heatmap = np.expand_dims(heatmap, axis=2)
        heatmap = np.tile(heatmap, [1, 1, 3])

        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap = cv2.COLORMAP_VIRIDIS):
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1-alpha, 0)

        return (heatmap, output)
