import tensorflow as tf
from grad_cam import GradCAM
from input_pipeline.preprocessing import preprocess
from guided_backprop import GuidedBackprop, deprocess_image
from matplotlib import pyplot as plt
import numpy as np
import os

import cv2

image_paths = ["/home/data/IDRID_dataset/images/train/IDRiD_001.jpg",
			   "/home/data/IDRID_dataset/images/train/IDRiD_002.jpg"]

def visualize(model, layername, save_path):
	j = 1
	for image_path in image_paths:
		fig = plt.figure()

		# process original image
		image = tf.io.read_file(image_path)
		image, temp = preprocess(image, None, 256, 256)
		image = np.expand_dims(image, axis=0)
		original_image = cv2.imread(image_path)
		original_image = original_image[0:2848, 230:3710]
		original_image = cv2.copyMakeBorder(original_image, 316, 316, 0, 0,
											cv2.BORDER_CONSTANT, value=[0,0,0])
		original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

		fig.add_subplot(2, 2, 1)
		plt.imshow(original_image)
		plt.axis('off')
		plt.title("original image")

		# use the network to make predictions on the input image and find
		# the class label index with the largest corresponding probability
		predictions = model.predict(image)
		i = np.argmax(predictions[0])

		# apply GradCAM
		# initialize our gradient class activation map and build the heatmap
		gradcam = GradCAM(model, i, layername)
		heatmap = gradcam.compute_heatmap(image)

		# resize the resulting heatmap to the original input image dimensions
		heatmap = cv2.resize(heatmap, (3480, 3480))
		(heatmap, gradcam_output) = gradcam.overlay_heatmap(heatmap, original_image, alpha=0.5)
		fig.add_subplot(2, 2, 2)
		plt.imshow(heatmap)
		plt.axis('off')
		plt.title("GradCAM")


		# apply Guided Backpropagation
		guidedback = GuidedBackprop(model, layername)
		gb = guidedback.guided_backpropagation(image)
		gb_show = deprocess_image(np.array(gb))

		fig.add_subplot(2, 2, 3)
		plt.imshow(gb_show)
		plt.axis('off')
		plt.title("Guided Backpropagation")

		# apply the Guided GradCAM
		gb_apply = np.maximum(gb, 0)
		gb_apply = gb_apply / np.max(gb_apply)
		gb_apply = gb_apply * 255
		gb_apply = gb_apply.astype(np.int16)

		guided_gradcam = gb_apply * heatmap
		fig.add_subplot(2, 2, 4)
		plt.imshow(guided_gradcam)
		plt.axis('off')
		plt.title("Guided GradCAM")
		plt.savefig(os.path.join(save_path, 'visualization' + str(j) + '.png'), bbox_inches='tight')
		j += 1