

import tensorflow as tf
from tensorflow.keras import layers, models, Input


def ResidualBlock(x, filters):
    """Residual blocks"""
    shortcut = x

    if x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)

    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([shortcut, x])
    return layers.Activation('relu')(x)

