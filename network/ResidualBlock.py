

import tensorflow as tf
from tensorflow.keras import layers, models, Input


def ResidualBlock(x, filters):
    """标准的残差模块 (Residual blocks)"""
    shortcut = x
    # 如果通道不一致，调整 shortcut 通道数
    if x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)

    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([shortcut, x])
    return layers.Activation('relu')(x)

