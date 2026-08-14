

"""
    Implementation of TransUnet
    @date 2026.06.10
    @author Dong Zhu

"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="TransUNet")
class PositionEmbedding(layers.Layer):
    """可训练的位置编码。"""

    def __init__(self, num_tokens, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.position_embedding = self.add_weight(
            name="position_embedding",
            shape=(1, self.num_tokens, self.embed_dim),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.position_embedding

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_tokens": self.num_tokens,
            "embed_dim": self.embed_dim,
        })
        return config


def conv_block(x, filters, dropout_rate=0.0, name="conv_block"):
    """U-Net 的双卷积模块。"""

    for index in range(1, 3):
        x = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            name=f"{name}_conv{index}",
        )(x)
        x = layers.BatchNormalization(
            name=f"{name}_bn{index}",
        )(x)
        x = layers.Activation(
            "relu",
            name=f"{name}_relu{index}",
        )(x)

        if dropout_rate > 0 and index == 1:
            x = layers.SpatialDropout2D(
                dropout_rate,
                name=f"{name}_dropout",
            )(x)

    return x


def transformer_block(
    x,
    embed_dim,
    num_heads,
    mlp_dim,
    dropout_rate=0.1,
    name="transformer",
):
    """Pre-Norm Transformer Encoder Block。"""

    # Multi-Head Self-Attention
    residual = x

    x = layers.LayerNormalization(
        epsilon=1e-6,
        name=f"{name}_attention_norm",
    )(x)

    x = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim // num_heads,
        dropout=dropout_rate,
        kernel_initializer="glorot_uniform",
        name=f"{name}_attention",
    )(x, x)

    x = layers.Dropout(
        dropout_rate,
        name=f"{name}_attention_dropout",
    )(x)

    x = layers.Add(
        name=f"{name}_attention_add",
    )([residual, x])

    # Feed Forward Network
    residual = x

    x = layers.LayerNormalization(
        epsilon=1e-6,
        name=f"{name}_mlp_norm",
    )(x)

    x = layers.Dense(
        mlp_dim,
        activation="gelu",
        name=f"{name}_mlp_dense1",
    )(x)
    x = layers.Dropout(
        dropout_rate,
        name=f"{name}_mlp_dropout1",
    )(x)

    x = layers.Dense(
        embed_dim,
        name=f"{name}_mlp_dense2",
    )(x)
    x = layers.Dropout(
        dropout_rate,
        name=f"{name}_mlp_dropout2",
    )(x)

    return layers.Add(
        name=f"{name}_mlp_add",
    )([residual, x])


def decoder_block(x, skip, filters, dropout_rate=0.0, name="decoder"):
    """上采样并融合 CNN 编码器的跳跃特征。"""

    x = layers.Conv2DTranspose(
        filters=filters,
        kernel_size=2,
        strides=2,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        name=f"{name}_upsample",
    )(x)
    x = layers.BatchNormalization(
        name=f"{name}_upsample_bn",
    )(x)
    x = layers.Activation(
        "relu",
        name=f"{name}_upsample_relu",
    )(x)

    x = layers.Concatenate(
        axis=-1,
        name=f"{name}_concat",
    )([x, skip])

    return conv_block(
        x,
        filters=filters,
        dropout_rate=dropout_rate,
        name=f"{name}_features",
    )


def TransUnet(
    image_size=128,
    input_channels=3,
    output_channels=1,
    base_filters=32,
    embed_dim=512,
    transformer_depth=12,
    num_heads=8,
    mlp_dim=1024,
    dropout_rate=0.1,
    output_activation=None,
):
    """
    用于点对点连续值预测的轻量 TransUNet。

    output_activation:
        None      -> 无范围限制的连续值预测
        "sigmoid" -> [0, 1] 范围内的连续值预测
        "tanh"    -> [-1, 1] 范围内的连续值预测
    """

    if image_size % 16 != 0:
        raise ValueError("image_size 必须能被 16 整除")

    if embed_dim % num_heads != 0:
        raise ValueError("embed_dim 必须能被 num_heads 整除")

    if transformer_depth <= 0:
        raise ValueError("transformer_depth 必须是正整数")

    inputs = keras.Input(
        shape=(image_size, image_size, input_channels),
        name="images",
    )

    # 如果输入是 0~255，可在这里归一化：
    # x = layers.Rescaling(1.0 / 255.0)(inputs)
    x = inputs

    encoder_filters = [
        base_filters,
        base_filters * 2,
        base_filters * 4,
        base_filters * 8,
    ]
    encoder_dropouts = [0.0, 0.0, 0.05, 0.1]
    skips = []

    # CNN Encoder: 128 -> 64 -> 32 -> 16 -> 8
    for index, filters in enumerate(encoder_filters):
        x = conv_block(
            x,
            filters=filters,
            dropout_rate=encoder_dropouts[index],
            name=f"encoder{index + 1}",
        )
        skips.append(x)

        x = layers.MaxPooling2D(
            pool_size=2,
            strides=2,
            padding="same",
            name=f"encoder{index + 1}_pool",
        )(x)

    feature_size = image_size // 16
    num_tokens = feature_size * feature_size

    # 将 CNN 特征投影到 Transformer 的 embed_dim。
    x = layers.Conv2D(
        filters=embed_dim,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name="transformer_projection",
    )(x)
    x = layers.BatchNormalization(
        name="transformer_projection_bn",
    )(x)
    x = layers.Activation(
        "relu",
        name="transformer_projection_relu",
    )(x)

    # [B, 8, 8, 256] -> [B, 64, 256]
    x = layers.Reshape(
        (num_tokens, embed_dim),
        name="feature_map_to_tokens",
    )(x)

    x = PositionEmbedding(
        num_tokens=num_tokens,
        embed_dim=embed_dim,
        name="position_embedding",
    )(x)

    x = layers.Dropout(
        dropout_rate,
        name="embedding_dropout",
    )(x)

    for index in range(transformer_depth):
        x = transformer_block(
            x,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
            name=f"transformer{index + 1}",
        )

    x = layers.LayerNormalization(
        epsilon=1e-6,
        name="transformer_output_norm",
    )(x)

    # [B, 64, 256] -> [B, 8, 8, 256]
    x = layers.Reshape(
        (feature_size, feature_size, embed_dim),
        name="tokens_to_feature_map",
    )(x)

    # U-Net Decoder: 8 -> 16 -> 32 -> 64 -> 128
    decoder_filters = list(reversed(encoder_filters))
    decoder_dropouts = [0.1, 0.05, 0.0, 0.0]

    for index, (filters, skip) in enumerate(
        zip(decoder_filters, reversed(skips))
    ):
        x = decoder_block(
            x,
            skip=skip,
            filters=filters,
            dropout_rate=decoder_dropouts[index],
            name=f"decoder{index + 1}",
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
        name="transunet_regression",
    )

    model.summary()

    return model


def main():
    model = TransUnet(
        image_size=128,
        input_channels=3,
        output_channels=1,
        base_filters=32,
        embed_dim=512,
        transformer_depth=6,
        num_heads=8,
        mlp_dim=1024,
        dropout_rate=0.1,
        output_activation=None,
    )

    model.summary()
    print(f"模型参数量: {model.count_params():,}")


if __name__ == "__main__":
    main()
