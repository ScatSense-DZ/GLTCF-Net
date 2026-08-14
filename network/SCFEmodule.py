

"""
    Implementation of Spatial-Channel Feature Refinement Module
    @date 2026.06.06
    @author Dong Zhu

"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import Multiply, Add, DepthwiseConv2D, Concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def get_scharr_initializer(channels, axis='x'):
    """生成 Scharr 算子权重"""
    if axis == 'x':
        kernel = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32)
    else:
        kernel = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float32)
    kernel = np.expand_dims(kernel, axis=-1)
    kernel = np.expand_dims(kernel, axis=-1)
    return tf.keras.initializers.Constant(np.tile(kernel, (1, 1, channels, 1)))


def SCFRM(inputs, reduction=4, name_prefix=""):
    """
    复杂版：Scharr 引导的双路联合注意力机制 (Spatial & Channel Dual Attention)
    """

    prefix = f"{name_prefix}_" if name_prefix else ""

    in_channels = inputs.shape[-1]

    # ==========================================
    # Phase 1: 物理高频先验提取 (Scharr Magnitude)
    # ==========================================
    grad_x = DepthwiseConv2D(3, padding='same', use_bias=False,
                             depthwise_initializer=get_scharr_initializer(in_channels, 'x'),
                             trainable=False, name=f'{prefix}scharr_x')(inputs)

    grad_y = DepthwiseConv2D(3, padding='same', use_bias=False,
                             depthwise_initializer=get_scharr_initializer(in_channels, 'y'),
                             trainable=False, name=f'{prefix}scharr_y')(inputs)

    # 计算梯度幅值
    # hf_prior = Lambda(lambda tensors: tf.sqrt(tf.square(tensors[0]) + tf.square(tensors[1]) + 1e-8),
    #                   name=f'{name}_magnitude')([grad_x, grad_y])

    # hf_prior = Lambda(lambda tensors: tf.abs(tensors[0]) + tf.abs(tensors[1]))([grad_x, grad_y])
    # hf_prior = Lambda(lambda tensors: K.abs(tensors[0]) + K.abs(tensors[1]))([grad_x, grad_y])
    hf_prior = tf.abs(grad_x) + tf.abs(grad_y)
    hf_prior = BatchNormalization(name=f'{prefix}bn_prior')(hf_prior)

    # ==========================================
    # Phase 2: 分支 1 —— 通道高频注意力 (Channel Stream)
    # “哪些通道对边缘最敏感？”
    # ==========================================
    # 同时使用 Average 和 Max Pooling，捕捉平均高频和极致高频
    avg_pool = GlobalAveragePooling2D(name=f'{prefix}ca_avg')(hf_prior)
    max_pool = GlobalMaxPooling2D(name=f'{prefix}ca_max')(hf_prior)

    # Reshape回4D以便使用 1x1 卷积(MLP)
    avg_pool = Reshape((1, 1, in_channels), name=f'{prefix}ca_reshape_avg')(avg_pool)
    max_pool = Reshape((1, 1, in_channels), name=f'{prefix}ca_reshape_max')(max_pool)

    # 共享权重的 MLP
    mlp_reduce = Conv2D(max(1, in_channels // reduction), 1, use_bias=False,
                               activation='relu', name=f'{prefix}ca_reduce')
    mlp_expand = Conv2D(in_channels, 1, use_bias=False, name=f'{prefix}ca_expand')

    avg_out = mlp_expand(mlp_reduce(avg_pool))
    max_out = mlp_expand(mlp_reduce(max_pool))

    # 融合并激活
    channel_weight = Add(name=f'{prefix}ca_add')([avg_out, max_out])
    channel_weight = Activation('sigmoid', name=f'{prefix}ca_sigmoid')(channel_weight)

    # 将通道注意力作用于原始特征
    ca_features = Multiply(name=f'{prefix}ca_multiply')([inputs, channel_weight])

    # ==========================================
    # Phase 3: 分支 2 —— 空间高频注意力 (Spatial Stream)
    # “图像的哪些像素是核心边缘？”
    # ==========================================
    # 沿通道维度取最大值和平均值，压缩为 2 个通道的空间图
    spatial_avg = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True), name=f'{prefix}sa_avg')(hf_prior)
    spatial_max = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True), name=f'{prefix}sa_max')(hf_prior)

    spatial_concat = Concatenate(axis=-1, name=f'{prefix}sa_concat')([spatial_avg, spatial_max])

    # 使用 7x7 大感受野卷积生成空间掩码
    spatial_weight = Conv2D(1, kernel_size=7, padding='same', use_bias=False,
                            activation='sigmoid', name=f'{prefix}sa_conv7x7')(spatial_concat)

    # 将空间注意力串联作用于通道增强后的特征
    sa_features = Multiply(name=f'{prefix}sa_multiply')([ca_features, spatial_weight])

    # ==========================================
    # Phase 4: 自适应残差融合 (Adaptive Residual)
    # ==========================================
    fused = Concatenate(name=f'{prefix}fusion_concat')([inputs, sa_features])
    out = Conv2D(in_channels, 1, padding='same', use_bias=False, name=f'{prefix}fusion_1x1')(fused)
    out = BatchNormalization(name=f'{prefix}fusion_bn')(out)

    # 主干捷径 (Short-cut)
    out = Add(name=f'{prefix}final_residual')([inputs, out])

    return out


# ==========================================
# 模块测试
# ==========================================
if __name__ == "__main__":
    input_tensor = Input(shape=(128, 128, 64))
    output_tensor = SCFRM(input_tensor)

    model = Model(inputs=input_tensor, outputs=output_tensor, name="Complex_Scharr_Attention")
    model.summary()
