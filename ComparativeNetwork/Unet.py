

"""
    Implementation of Unet
    @date 2026.06.10
    @author Dong Zhu

"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def conv_block(x, filters, dropout_rate=0.0, name="conv_block"):
    x = layers.Conv2D(
        filters=filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name}_conv1",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)

    if dropout_rate > 0:
        x = layers.SpatialDropout2D(
            rate=dropout_rate,
            name=f"{name}_dropout",
        )(x)

    x = layers.Conv2D(
        filters=filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name}_conv2",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)

    return x


def encoder_block(
    x,
    filters,
    dropout_rate=0.0,
    name="encoder",
):

    skip = conv_block(
        x,
        filters=filters,
        dropout_rate=dropout_rate,
        name=f"{name}_features",
    )

    pooled = layers.MaxPooling2D(
        pool_size=2,
        strides=2,
        padding="same",
        name=f"{name}_pool",
    )(skip)

    return skip, pooled


def decoder_block(
    x,
    skip,
    filters,
    dropout_rate=0.0,
    name="decoder",
):

    x = layers.Conv2DTranspose(
        filters=filters,
        kernel_size=2,
        strides=2,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name}_upsample",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_upsample_bn")(x)
    x = layers.Activation("relu", name=f"{name}_upsample_relu")(x)

    x = layers.Concatenate(
        axis=-1,
        name=f"{name}_concat",
    )([x, skip])

    x = conv_block(
        x,
        filters=filters,
        dropout_rate=dropout_rate,
        name=f"{name}_features",
    )

    return x


def Unet(
    image_size=128,
    input_channels=3,
    output_channels=1,
    base_filters=32,
    output_activation=None,
):

    if image_size % 16 != 0:
        raise ValueError("image_size 必须能被 16 整除")

    if base_filters <= 0:
        raise ValueError("base_filters 必须是正整数")

    inputs = keras.Input(
        shape=(image_size, image_size, input_channels),
        name="images",
    )

    x = inputs

    # 128x128
    skip1, x = encoder_block(
        x,
        filters=base_filters,
        dropout_rate=0.0,
        name="encoder1",
    )

    # 64x64
    skip2, x = encoder_block(
        x,
        filters=base_filters * 2,
        dropout_rate=0.0,
        name="encoder2",
    )

    # 32x32
    skip3, x = encoder_block(
        x,
        filters=base_filters * 4,
        dropout_rate=0.05,
        name="encoder3",
    )

    # 16x16
    skip4, x = encoder_block(
        x,
        filters=base_filters * 8,
        dropout_rate=0.1,
        name="encoder4",
    )

    # Bottleneck: 8x8
    x = conv_block(
        x,
        filters=base_filters * 16,
        dropout_rate=0.2,
        name="bottleneck",
    )

    # 8x8 -> 16x16
    x = decoder_block(
        x,
        skip=skip4,
        filters=base_filters * 8,
        dropout_rate=0.1,
        name="decoder4",
    )

    # 16x16 -> 32x32
    x = decoder_block(
        x,
        skip=skip3,
        filters=base_filters * 4,
        dropout_rate=0.05,
        name="decoder3",
    )

    # 32x32 -> 64x64
    x = decoder_block(
        x,
        skip=skip2,
        filters=base_filters * 2,
        dropout_rate=0.0,
        name="decoder2",
    )

    # 64x64 -> 128x128
    x = decoder_block(
        x,
        skip=skip1,
        filters=base_filters,
        dropout_rate=0.0,
        name="decoder1",
    )

    outputs = layers.Conv2D(
        filters=output_channels,
        kernel_size=1,
        padding="same",
        activation=output_activation,
        dtype="float32",
        name="prediction",
    )(x)
    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="unet_regression",
    )

    model.summary()

    return model


def main():
    model = Unet(
        image_size=128,
        input_channels=3,
        output_channels=1,
        base_filters=32,
        output_activation=None,
    )

    model.summary()
    print(f"参数量: {model.count_params():,}")


if __name__ == "__main__":
    main()
