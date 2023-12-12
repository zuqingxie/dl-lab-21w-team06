import gin
import tensorflow as tf
import random
import tensorflow_addons as tfa
import numpy as np

@gin.configurable
def preprocess(image, label, img_height, img_width):
    # Resize image
    image = tf.image.decode_jpeg(image)
    image = tf.image.crop_to_bounding_box(image,0,230,2848,3480)
    image = tf.image.resize_with_pad(image, img_height, img_width)
    image = image.numpy()

    return image, label

def normalize(image, label):
    img_max = tf.reduce_max(image)
    image = tf.cast(image, tf.float32) / tf.cast(img_max, tf.float32)
    return image, label

@gin.configurable
def raw_or_aug(image, func, i=0.4):
    if random.uniform(0, 1) < i:
        image = func(image)
    else:
        image = image
    return image

def random_brightness(image):
    return tf.image.stateless_random_brightness(image, max_delta=0.3, seed=(random.randint(1,10000), 0))

def random_contrast(image):
    return tf.image.stateless_random_contrast(image, lower=0.5, upper=1.5, seed=(random.randint(1,10000), 0))

def random_flip_left_right(image):
    return tf.image.stateless_random_flip_left_right(image, seed=(random.randint(1,10000), 0))

def random_flip_up_down(image):
    return tf.image.stateless_random_flip_up_down(image,seed=(random.randint(1,10000), 0))

def random_rotate(image):
    # rotate randomly between +90° and -90°
    image = tfa.image.rotate(image, np.random.uniform(-np.pi/2, np.pi/2))
    return image

def random_crop(image):
    factor = random.uniform(0.8, 1)
    height = int(256 * factor)
    width = int(256 * factor)
    image = tf.image.stateless_random_crop(image, (height, width, 3), seed=(random.randint(1, 10000), 0))
    image = tf.image.resize(image, [256, 256])
    return image


def augment(image, label):
    """Data augmentation"""
    image = raw_or_aug(image, random_brightness)
    image = raw_or_aug(image, random_contrast)
    image = raw_or_aug(image, random_flip_left_right)
    image = raw_or_aug(image, random_flip_up_down)
    image = raw_or_aug(image, random_rotate)
    image = raw_or_aug(image, random_crop)

    return image, label


