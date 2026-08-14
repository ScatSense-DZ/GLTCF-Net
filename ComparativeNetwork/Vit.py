
"""
    Implementation of ViT
    @date 2026.06.14
    @author Dong Zhu

"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class PatchEmbedding(layers.Layer):

    def __init__(self, image_size, patch_size, embed_dim, **kwargs):
        super().__init__(**kwargs)

        if image_size % patch_size != 0:
            raise ValueError("image_size 必须能被 patch_size 整除")

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2

        self.projection = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            name="projection",
        )

    def build(self, input_shape):
        self.position_embedding = self.add_weight(
            name="position_embedding",
            shape=(1, self.num_patches, self.embed_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, images):
        # [B, 128, 128, 3] -> [B, 16, 16, embed_dim]
        patches = self.projection(images)

        batch_size = tf.shape(patches)[0]

        # [B, 16, 16, embed_dim] -> [B, 256, embed_dim]
        tokens = tf.reshape(
            patches,
            [batch_size, self.num_patches, self.embed_dim],
        )

        return tokens + self.position_embedding

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config


class MLP(layers.Layer):
    def __init__(
        self,
        hidden_dim,
        output_dim,
        dropout_rate=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        self.dense1 = layers.Dense(hidden_dim, activation=tf.nn.gelu)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(output_dim)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training=None):
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        return self.dropout2(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class TransformerEncoder(layers.Layer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_dim,
        dropout_rate=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能被 num_heads 整除")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate,
        )
        self.dropout1 = layers.Dropout(dropout_rate)

        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = MLP(
            hidden_dim=mlp_dim,
            output_dim=embed_dim,
            dropout_rate=dropout_rate,
        )

    def call(self, x, training=None):
        # Pre-Norm Multi-Head Self-Attention
        normalized = self.norm1(x)
        attention_output = self.attention(
            query=normalized,
            value=normalized,
            key=normalized,
            training=training,
        )
        attention_output = self.dropout1(
            attention_output,
            training=training,
        )
        x = x + attention_output

        # Pre-Norm Feed Forward Network
        x = x + self.mlp(
            self.norm2(x),
            training=training,
        )

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


def decoder_block(x, filters, name):

    x = layers.Conv2DTranspose(
        filters=filters,
        kernel_size=2,
        strides=2,
        padding="same",
        use_bias=False,
        name=f"{name}_upsample",
    )(x)
    x = layers.LayerNormalization(
        epsilon=1e-6,
        name=f"{name}_norm1",
    )(x)
    x = layers.Activation(
        tf.nn.gelu,
        name=f"{name}_act1",
    )(x)

    residual = x

    x = layers.Conv2D(
        filters=filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name=f"{name}_conv1",
    )(x)
    x = layers.LayerNormalization(
        epsilon=1e-6,
        name=f"{name}_norm2",
    )(x)
    x = layers.Activation(
        tf.nn.gelu,
        name=f"{name}_act2",
    )(x)

    x = layers.Conv2D(
        filters=filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name=f"{name}_conv2",
    )(x)
    x = layers.LayerNormalization(
        epsilon=1e-6,
        name=f"{name}_norm3",
    )(x)

    x = layers.Add(name=f"{name}_residual")([x, residual])
    x = layers.Activation(
        tf.nn.gelu,
        name=f"{name}_output",
    )(x)

    return x


def Vit(
    image_size=128,
    patch_size=8,
    input_channels=3,
    output_channels=1,
    embed_dim=512,
    depth=12,
    num_heads=8,
    mlp_dim=512,
    dropout_rate=0.1,
):

    if image_size % patch_size != 0:
        raise ValueError("image_size 必须能被 patch_size 整除")

    if patch_size <= 0 or patch_size & (patch_size - 1):
        raise ValueError("patch_size 必须是 2 的正整数次幂")

    if embed_dim % num_heads != 0:
        raise ValueError("embed_dim 必须能被 num_heads 整除")

    grid_size = image_size // patch_size

    inputs = keras.Input(
        shape=(image_size, image_size, input_channels),
        name="images",
    )

    x = inputs

    # [B, 128, 128, 3] -> [B, 256, 512]
    x = PatchEmbedding(
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        name="patch_embedding",
    )(x)

    for index in range(depth):
        x = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate,
            name=f"transformer_encoder_{index}",
        )(x)

    x = layers.LayerNormalization(
        epsilon=1e-6,
        name="encoder_norm",
    )(x)

    # [B, 256, 512] -> [B, 16, 16, 512]
    x = layers.Reshape(
        target_shape=(grid_size, grid_size, embed_dim),
        name="tokens_to_feature_map",
    )(x)

    # 16x16 -> 32x32
    x = decoder_block(
        x,
        filters=256,
        name="decoder_0",
    )

    # 32x32 -> 64x64
    x = decoder_block(
        x,
        filters=128,
        name="decoder_1",
    )

    # 64x64 -> 128x128
    x = decoder_block(
        x,
        filters=64,
        name="decoder_2",
    )

    x = layers.Conv2D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation=tf.nn.gelu,
        name="prediction_conv",
    )(x)

    outputs = layers.Conv2D(
        filters=output_channels,
        kernel_size=1,
        padding="same",
        activation=None,
        dtype="float32",
        name="prediction",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="dense_regression_vit",
    )

    model.summary()

    return model


def main():
    model = Vit()

    model.summary()

    print(f"模型参数量: {model.count_params():,}")


if __name__ == "__main__":
    main()
