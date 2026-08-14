
"""
    Implementation of Anisotropic Feature Integration Module
    @date 2026.05.22
    @author Dong Zhu

"""

import tensorflow as tf
from tensorflow.keras import layers


def window_partition(x, window_size):

    input_shape = tf.shape(x)
    H, W = input_shape[1], input_shape[2]
    C = x.shape[-1]

    x = tf.reshape(x, [-1, H // window_size, window_size, W // window_size, window_size, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, [-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, H, W, C):
    x = tf.reshape(windows, [-1, H // window_size, W // window_size, window_size, window_size, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [-1, H, W, C])
    return x


class WindowAttention(layers.Layer):
    def __init__(self, dim, num_heads, window_size, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads)

    def call(self, x):

        C = self.dim
        W = self.window_size

        x_flat = tf.reshape(x, [-1, W * W, C])

        attn_out = self.attn(x_flat, x_flat, x_flat)
        out = tf.reshape(attn_out, [-1, W, W, C])
        return out


class SwinTransformerBlock(layers.Layer):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        self.mlp = tf.keras.Sequential([
            layers.Dense(dim * 4, activation='gelu'),
            layers.Dense(dim)
        ])

    def call(self, x):
        input_shape = tf.shape(x)
        H, W = input_shape[1], input_shape[2]
        C = self.dim
        shortcut = x

        x = self.norm1(x)

        if self.shift_size > 0:
            shifted_x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)

        attn_windows = self.attn(x_windows)

        shifted_x_out = window_reverse(attn_windows, self.window_size, H, W, C)

        if self.shift_size > 0:
            x = tf.roll(shifted_x_out, shift=[self.shift_size, self.shift_size], axis=[1, 2])
        else:
            x = shifted_x_out

        x = x + shortcut
        x = x + self.mlp(self.norm2(x))
        return x

    def get_config(self):

        config = super(SwinTransformerBlock, self).get_config()

        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "window_size": self.window_size,
            "shift_size": self.shift_size,
        })
        return config


def AFIM(input_tensor, filters, num_heads=4, window_size=8, k=3, name_prefix=''):

    prefix = f"{name_prefix}_" if name_prefix else ""

    x_trans = SwinTransformerBlock(dim=filters,
                                   num_heads=num_heads,
                                   window_size=window_size,
                                   shift_size=0,
                                   name=f'{prefix}w_trans')(input_tensor)

    x_trans = SwinTransformerBlock(dim=filters,
                                   num_heads=num_heads,
                                   window_size=window_size,
                                   shift_size=window_size // 2,
                                   name=f'{prefix}sw_trans')(x_trans)

    x_cnn = input_tensor

    x_cnn = layers.Conv2D(filters, kernel_size=(3, 3),
                          padding='same', dilation_rate=2,
                          activation='gelu',
                          name=f'{prefix}conv_dilated_3x3')(x_cnn)

    conv_h = layers.Conv2D(filters, kernel_size=(1, k), padding='same', name=f'{prefix}conv_h_1x{k}')(x_cnn)
    conv_c = layers.Conv2D(filters, kernel_size=(k, k), padding='same', name=f'{prefix}conv_c_{k}x{k}')(x_cnn)
    conv_v = layers.Conv2D(filters, kernel_size=(k, 1), padding='same', name=f'{prefix}conv_v_{k}x1')(x_cnn)

    cnn_merged = layers.Add(name=f'{prefix}add_parallel_convs')([conv_h, conv_c, conv_v])
    cnn_merged = layers.Activation('gelu', name=f'{prefix}gelu')(cnn_merged)

    cnn_out = layers.Conv2D(filters, kernel_size=(1, 1), padding='same', name=f'{prefix}conv_1x1')(cnn_merged)

    output = layers.Add(name=f'{prefix}final_fusion_add')([x_trans, cnn_out])

    return output


# if __name__ == "__main__":
#     B, H, W, C = 2, 128, 128, 64
#
#     inputs = tf.keras.Input(shape=(H, W, C))
#     outputs = AFIM(inputs, filters=C, num_heads=4, window_size=8)
#
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#
#     model.summary()
