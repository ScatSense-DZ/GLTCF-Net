

"""
    Implementation of ResNet
    @date 2026.06.05
    @author Dong Zhu

"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def conv_bn_act(
    x,
    filters,
    kernel_size,
    strides=1,
    activation=True,
    name=None,
):
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=None if name is None else f"{name}_conv",
    )(x)

    x = layers.BatchNormalization(
        name=None if name is None else f"{name}_bn",
    )(x)

    if activation:
        x = layers.Activation(
            "relu",
            name=None if name is None else f"{name}_relu",
        )(x)

    return x


def basic_block(x, filters, strides=1, name="block"):
    """ResNet18/34 使用的 Basic Block。"""

    shortcut = x

    if strides != 1 or x.shape[-1] != filters:
        shortcut = conv_bn_act(
            shortcut,
            filters=filters,
            kernel_size=1,
            strides=strides,
            activation=False,
            name=f"{name}_shortcut",
        )

    x = conv_bn_act(
        x,
        filters=filters,
        kernel_size=3,
        strides=strides,
        name=f"{name}_conv1",
    )

    x = conv_bn_act(
        x,
        filters=filters,
        kernel_size=3,
        activation=False,
        name=f"{name}_conv2",
    )

    x = layers.Add(name=f"{name}_add")([x, shortcut])
    return layers.Activation("relu", name=f"{name}_output")(x)


def make_resnet_stage(x, filters, blocks, first_stride, name):
    x = basic_block(
        x,
        filters=filters,
        strides=first_stride,
        name=f"{name}_block1",
    )

    for index in range(2, blocks + 1):
        x = basic_block(
            x,
            filters=filters,
            name=f"{name}_block{index}",
        )

    return x


def decoder_block(x, skip, filters, name):
    """上采样并融合编码器特征。"""

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

    if skip is not None:
        skip = conv_bn_act(
            skip,
            filters=filters,
            kernel_size=1,
            name=f"{name}_skip",
        )

        x = layers.Concatenate(name=f"{name}_concat")([x, skip])

        x = conv_bn_act(
            x,
            filters=filters,
            kernel_size=3,
            name=f"{name}_fusion",
        )

    # 每层只保留一个残差块，进一步降低参数量。
    return basic_block(
        x,
        filters=filters,
        name=f"{name}_residual",
    )


def ResNet18(
    image_size=128,
    input_channels=3,
    output_channels=1,
    decoder_filters=(256, 128, 64, 32, 16),
    output_activation=None,
):
    """
    ResNet18 编码器 + 轻量残差解码器。

    输入: [B, 128, 128, 3]
    输出: [B, 128, 128, 1]
    """

    if image_size % 32 != 0:
        raise ValueError("image_size 必须能被 32 整除")

    if len(decoder_filters) != 5:
        raise ValueError("decoder_filters 必须包含 5 个整数")

    inputs = keras.Input(
        shape=(image_size, image_size, input_channels),
        name="images",
    )

    # 输入是否归一化应和数据加载流程保持一致。
    x = inputs

    # 128x128 -> 64x64
    c1 = conv_bn_act(
        x,
        filters=64,
        kernel_size=7,
        strides=2,
        name="stem",
    )

    # 64x64 -> 32x32
    x = layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding="same",
        name="stem_pool",
    )(c1)

    # ResNet18 的 block 数量为 [2, 2, 2, 2]。

    # 32x32，64 channels
    c2 = make_resnet_stage(
        x,
        filters=64,
        blocks=2,
        first_stride=1,
        name="stage2",
    )

    # 16x16，128 channels
    c3 = make_resnet_stage(
        c2,
        filters=128,
        blocks=2,
        first_stride=2,
        name="stage3",
    )

    # 8x8，256 channels
    c4 = make_resnet_stage(
        c3,
        filters=256,
        blocks=2,
        first_stride=2,
        name="stage4",
    )

    # 4x4，512 channels
    c5 = make_resnet_stage(
        c4,
        filters=512,
        blocks=2,
        first_stride=2,
        name="stage5",
    )

    # 4x4 -> 8x8
    x = decoder_block(
        c5,
        skip=c4,
        filters=decoder_filters[0],
        name="decoder1",
    )

    # 8x8 -> 16x16
    x = decoder_block(
        x,
        skip=c3,
        filters=decoder_filters[1],
        name="decoder2",
    )

    # 16x16 -> 32x32
    x = decoder_block(
        x,
        skip=c2,
        filters=decoder_filters[2],
        name="decoder3",
    )

    # 32x32 -> 64x64
    x = decoder_block(
        x,
        skip=c1,
        filters=decoder_filters[3],
        name="decoder4",
    )

    # 64x64 -> 128x128
    x = decoder_block(
        x,
        skip=None,
        filters=decoder_filters[4],
        name="decoder5",
    )

    x = conv_bn_act(
        x,
        filters=decoder_filters[-1],
        kernel_size=3,
        name="prediction_features",
    )

    # 点对点连续值预测，默认使用线性输出。
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
        name="dense_resnet18",
    )

    model.summary()

    return model


def main():
    model = ResNet18(
        image_size=128,
        input_channels=3,
        output_channels=1,
        decoder_filters=(256, 128, 64, 32, 16),
        output_activation=None,
    )

    model.summary()
    print(f"模型参数量: {model.count_params():,}")


if __name__ == "__main__":
    main()
