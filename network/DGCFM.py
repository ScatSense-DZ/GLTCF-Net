

"""
    Implementation of Difference-Guided Cross Fusion Module
    @date 2026.05.26
    @author Dong Zhu

"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, Multiply, Add, Activation, BatchNormalization, Lambda


def DGCFM(f_g, f_l, out_channels, name_prefix=""):

    prefix = f"{name_prefix}_" if name_prefix else ""

    # 1. 计算绝对差异
    diff = Lambda(lambda x: tf.abs(x[0] - x[1]), name=f'{prefix}abs_diff')([f_g, f_l])

    # ---------------------------------------------------------
    # 2. 差异聚合与掩码生成 (将 SeparableConv 改为标准 Conv2D)
    # ---------------------------------------------------------
    # --- 上支路 ---

    concat_g = Concatenate(axis=-1, name=f'{prefix}concat_g')([f_g, diff])
    conv_g = Conv2D(out_channels, kernel_size=3, padding='same', use_bias=False, name=f'{prefix}conv3x3_g')(concat_g)
    conv_g = BatchNormalization(name=f'{prefix}bn_g')(conv_g)
    conv_g = Activation('relu', name=f'{prefix}relu_g')(conv_g)
    mask_g = Conv2D(out_channels, kernel_size=1, padding='same', activation='sigmoid', name=f'{prefix}mask_g')(conv_g)

    # --- 下支路 ---
    concat_l = Concatenate(axis=-1, name=f'{prefix}concat_l')([f_l, diff])
    conv_l = Conv2D(out_channels, kernel_size=3, padding='same', use_bias=False, name=f'{prefix}conv3x3_l')(concat_l)
    conv_l = BatchNormalization(name=f'{prefix}bn_l')(conv_l)
    conv_l = Activation('relu', name=f'{prefix}relu_l')(conv_l)
    mask_l = Conv2D(out_channels, kernel_size=1, padding='same', activation='sigmoid', name=f'{prefix}mask_l')(conv_l)

    # 3. 交叉反馈调制 (核心逻辑不变)
    f_g_modulated = Multiply(name=f'{prefix}cross_mult_g')([f_g, mask_l])
    f_l_modulated = Multiply(name=f'{prefix}cross_mult_l')([f_l, mask_g])

    # 4. 残差补偿与最终融合
    out_g = Add(name=f'{prefix}res_add_g')([f_g, f_g_modulated])
    out_l = Add(name=f'{prefix}res_add_l')([f_l, f_l_modulated])

    final_concat = Concatenate(axis=-1, name=f'{prefix}final_concat')([out_g, out_l])

    # 最后的融合也替换为标准 3x3 卷积
    fused_out = Conv2D(out_channels, kernel_size=3, padding='same', use_bias=False, name=f'{prefix}final_conv3x3')(final_concat)
    fused_out = BatchNormalization(name=f'{prefix}final_bn')(fused_out)
    fused_out = Activation('relu', name=f'{prefix}final_relu')(fused_out)

    return fused_out


# --- 测试代码 ---
if __name__ == "__main__":
    from tensorflow.keras import Input, Model

    # 假设你的特征图大小为 64x64，通道数为 128
    channels = 128
    input_g = Input(shape=(64, 64, channels), name='global_input')
    input_l = Input(shape=(64, 64, channels), name='local_input')

    # 调用你写好的融合模块
    output = DGCF_module(input_g, input_l, out_channels=channels)

    # 实例化模型并打印 summary
    model = Model(inputs=[input_g, input_l], outputs=output)
    model.summary()

