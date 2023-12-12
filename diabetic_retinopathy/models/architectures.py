import gin
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, regularizers
from models.layers import vgg_block, res_block
from tensorflow.keras.applications import InceptionResNetV2, InceptionV3, MobileNetV2, Xception


@gin.configurable
def vgg_like(input_shape, n_classes, base_filters, n_blocks, dense_units, dropout_rate):
    """Defines a VGG-like architecture.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons
        base_filters (int): number of base filters, which are doubled for every VGG block
        n_blocks (int): number of VGG blocks
        dense_units (int): number of dense units
        dropout_rate (float): dropout rate

    Returns:
        (keras.Model): keras model object
    """

    assert n_blocks > 0, 'Number of blocks has to be at least 1.'

    inputs = tf.keras.Input(input_shape)
    out = vgg_block(inputs, base_filters)
    for i in range(2, n_blocks):
        out = vgg_block(out, base_filters * 2 ** (i))
    out = layers.GlobalAveragePooling2D()(out)
    out = layers.Dense(dense_units, activation=tf.nn.relu, kernel_regularizer=regularizers.L1L2(l1=0.01, l2=0.005))(out)
    out = layers.Dropout(dropout_rate)(out)
    outputs = layers.Dense(n_classes)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg_like')


@gin.configurable
def Transfer_module(num_classes, dropout_rate, module_name):
    """
    Define the transfer learning including InceptionResNetV2, InceptionV3, MobileNetV2 and Xception
    Args:
        num_classes: number of the class, here should always be 2.
        dropout_rate: dropout rate
        module_name: model name of the transfer learning

    Returns:
        (keras.Model): keras model object
    """
    if module_name == 'mobilenet':
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(256, 256, 3)
        )

    elif module_name == 'inceptionV3':
        base_model = InceptionV3(
            weights='imagenet',
            include_top=False,
            input_shape=(256, 256, 3)
        )
    elif module_name == 'inception_resnet_v2':
        base_model = InceptionResNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(256, 256, 3)
        )
    elif module_name == 'xception':
        base_model = Xception(
            weights='imagenet',
            include_top=False,
            input_shape=(256, 256, 3)
        )
    else:
        raise ValueError
    base_model.trainable = False
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(256, 256, 3)),
            base_model,
            layers.Conv2D(32, 3, padding="same"),
            layers.GlobalAvgPool2D(),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes)
        ]
    )

    return model


@gin.configurable
def resnet_simple(input_shape, block_num, filters, kernel_size, n_classes, dropout_rate, Dense_fc_num):
    """
    Define the transfer learning including InceptionResNetV2, InceptionV3, MobileNetV2 and Xception
    Args:
        input_shape: input shape
        block_num: number of the res_block
        filters: number of filters used for the convolutional layers
        kernel_size: kernel size
        n_classes: number of the classes, should be 2 only in this project
        dropout_rate: dropout rate
        Dense_fc_num: number of the full connection units in Dense layer

    Returns:
        (keras.Model): keras model object
    """

    inputs = tf.keras.Input(shape=input_shape)
    out = layers.Conv2D(filters, 3, 1, padding="same", activation="relu", kernel_regularizer=regularizers.l2(0.01))(inputs)
    out = layers.BatchNormalization()(out)
    out = layers.MaxPool2D(pool_size=(2, 2), strides=1, padding="same")(out)

    assert block_num > 0    # block_num must be bigger than 0
    for i in range(block_num):
        out = res_block(out, filters=filters*(2**(i)), kernel_size=kernel_size, strides=1)
        out = res_block(out, filters=filters*(2**(i)), kernel_size=kernel_size, strides=2)

    out = layers.Conv2D(filters*(2**(block_num)), 3, padding="same")(out)
    out = layers.GlobalAvgPool2D()(out)
    out = layers.Dense(Dense_fc_num, activation="relu", kernel_regularizer=regularizers.l1(0.01))(out)
    out = layers.Dropout(dropout_rate)(out)
    outputs = layers.Dense(n_classes)(out)
    return tf.keras.Model(inputs, outputs, name="resnet_simple")


@gin.configurable
def CNN_team06(input_shape, filters, n_classes, strides, kernel_size, pool_size, dropout_rate):
    """
    Custom CNN model from team06
    Args:
        input_shape: input shape
        filters: number of filters used for the convolutional layers
        n_classes: number of the classes, should be 2 only in this project
        strides: strides size in convolution
        kernel_size: kernel size
        pool_size: pool size in MaxPool2D layer
        dropout_rate: dropout rate

    Returns:
        (keras.Model): keras model object
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))

    Conv1 = layers.Conv2D(
        filters=filters[0], kernel_size=kernel_size[0],
        strides=strides[0], activation="relu", padding="same",
        kernel_regularizer=regularizers.L1(l1=0.01))
    model.add(Conv1)
    model.add(layers.MaxPool2D(pool_size=pool_size))
    model.add(layers.BatchNormalization())

    Conv2 = layers.Conv2D(
        filters=filters[1], kernel_size=kernel_size[1], padding="same",
        strides=strides[1], activation="relu",
        kernel_regularizer=regularizers.L1(l1=0.01))
    model.add(Conv2)

    model.add(layers.MaxPool2D(pool_size=pool_size))
    model.add(layers.BatchNormalization())

    Conv3 = layers.Conv2D(
        filters=filters[2], kernel_size=kernel_size[2],
        strides=strides[2], activation="relu",
        kernel_regularizer=regularizers.L1(l1=0.005))
    model.add(Conv3)

    model.add(layers.MaxPool2D(pool_size=pool_size))
    model.add(layers.BatchNormalization())

    Conv4 = layers.Conv2D(
        filters=filters[3], kernel_size=kernel_size[3],
        strides=strides[3], activation="relu",
        kernel_regularizer=regularizers.L1(l1=0.005))
    model.add(Conv4)

    model.add(tf.keras.layers.GlobalAveragePooling2D())

    model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(units=8, activation="relu", kernel_regularizer=regularizers.L1(l1=0.01)))

    model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(units=n_classes, kernel_regularizer=regularizers.L1L2(l1=0.01, l2=0.005)))

    return model
