

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from network.DGCFM import DGCFM
from network.ResidualBlock import ResidualBlock
from network.AFIM import AFIM
from network.SCFEmodule import SCFRM


def OverallNetwork(input_shape_A=(128, 128, 1), input_shape_B=(128, 128, 2), kernel_size=3):

    in_A = Input(shape=input_shape_A, name="Input_DEM_Continuous")
    in_B = Input(shape=input_shape_B, name="Input_Radar_Params")

    # 阶段通道数及模块重复次数设定
    # filters_list = [64, 128, 256, 512]
    filters_list = [32, 64, 128, 256]
    num_heads_list = [2, 4, 8, 16]
    repeats_list = [2, 2, 4, 2]  # 图中标识的 AFEM 和 ResBlock 的循环次数
    window_size = 8

    # --------------------------------------------------
    # 【输入预处理 / 下采样至 h/2】
    # --------------------------------------------------
    # 左侧：Linear Embedding (假设输入为 H, W，需下采样到 H/2)
    left_x = layers.Conv2D(filters_list[0], kernel_size=2, strides=2, padding='valid', name="Linear_Embedding")(in_A)

    # 中间：将输入也映射到 H/2, 64通道
    mid_x = layers.Conv2D(filters_list[0], kernel_size=3, strides=2, padding='same', name="Mid_Stem_Conv")(in_B)

    skip_connections = []  # 用于保存右侧解码器的跳跃连接

    # --------------------------------------------------
    # 【编码器 Encoder: Stage 1 到 Stage 4】
    # --------------------------------------------------
    for i in range(4):
        stage_filters = filters_list[i]

        # --- 如果不是 Stage 1，需先进行空间下采样 ---
        if i > 0:
            # 左侧 Transformer 分支下采样 (Patch Merging 替代方案：跨步卷积)
            left_x = layers.Conv2D(stage_filters, kernel_size=3, strides=2, padding='same')(left_x)
            # 中间 CNN 分支下采样
            mid_x = layers.Conv2D(stage_filters, kernel_size=3, strides=2, padding='same')(mid_x)

        # --- 左侧分支处理: AFE module ---
        num_AFE = repeats_list[i] if i != 2 else 4
        for j in range(num_AFE):  # Stage 3 的 AFEM 标了 x4
            left_x = AFIM(left_x, stage_filters, num_heads_list[i], window_size,
                                k=kernel_size, name_prefix=f"Stage{i + 1}_AFIM_{j + 1}")

        # --- 中间分支处理: Residual blocks + SCFE module ---
        for j in range(repeats_list[i]):
            mid_x = ResidualBlock(mid_x, stage_filters)
        mid_x = SCFRM(mid_x, stage_filters, name_prefix=f"Stage{i + 1}_SCE")

        # --- 跨分支融合:  DGCF module ---
        fused_feat = DGCFM(left_x, mid_x, stage_filters, name_prefix=f"Stage{i + 1}_DGCF")

        # 将 DGCF_module 的输出保存为 Skip Connection
        skip_connections.append(fused_feat)

        # DGCF_module 的输出同时作为中间分支下一阶段的输入 (覆盖 mid_x)
        mid_x = fused_feat
        # 左侧特征 left_x 直接流向下一阶段

    # skip_connections 里保存的特征分辨率依次为: [h/2, h/4, h/8, h/16]
    # 反转列表顺序，方便解码器从底向上提取: [h/16, h/8, h/4, h/2]
    skip_connections = skip_connections[::-1]

    # --------------------------------------------------
    # 【解码器 Decoder (Right Branch)】
    # --------------------------------------------------
    # 从 Stage 4 的输出开始 (对应 h/16)
    dec_x = skip_connections[0]

    # 依次向上层解码 (合并 h/8, h/4, h/2 的特征)
    for i in range(1, 4):
        decoder_filters = filters_list[4 - i - 1]  # 256 -> 128 -> 64

        # 1. 红线：TConv 3x3 上采样
        dec_x = layers.Conv2DTranspose(decoder_filters, kernel_size=3, strides=2, padding='same', name=f"TConv_{i}")(
            dec_x)

        # 2. 虚线框：Feature Concat (与左侧对应的紫框特征拼接)
        dec_x = layers.Concatenate(name=f"Concat_{i}")([dec_x, skip_connections[i]])

        # 3. 绿线：Conv 3x3, BN, ReLU
        dec_x = layers.Conv2D(decoder_filters, 3, padding='same')(dec_x)
        dec_x = layers.BatchNormalization()(dec_x)
        dec_x = layers.Activation('relu')(dec_x)

    # --------------------------------------------------
    # 【最后输出层 (从 h/2 恢复至 h, w 并输出 1 通道)】
    # --------------------------------------------------
    # 经过上面的循环，现在的 dec_x 分辨率是 h/2, w/2
    # 图顶部的最终预测是一个 h, w, 1 的图像，所以需要最后一次 TConv
    dec_x = layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', name="Final_TConv")(dec_x)
    dec_x = layers.Conv2D(1, 3, padding='same')(dec_x)
    dec_x = layers.BatchNormalization()(dec_x)
    dec_x = layers.Activation('relu')(dec_x)

    # 最后的绿线/输出通道
    final_output = layers.Conv2D(1, kernel_size=1, activation='linear', name="Radar_Clutter_Map")(dec_x)

    # 组装模型
    model = models.Model(inputs=[in_A, in_B], outputs=final_output, name="DualBranch_Transformer_Model")

    # 打印网络结构总结
    model.summary()

    return model


# ==========================================
# 4. 测试实例化与参数量查看
# ==========================================
if __name__ == "__main__":
    # 假设输入尺寸为 256x256
    model = OverallNetwork(input_shape_A=(128, 128, 1), input_shape_B=(128, 128, 2))


