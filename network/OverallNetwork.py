

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from network.DGCFM import DGCFM
from network.ResidualBlock import ResidualBlock
from network.AFIM import AFIM
from network.SCFEmodule import SCFRM


def OverallNetwork(input_shape_A=(128, 128, 1), input_shape_B=(128, 128, 2), kernel_size=3):

    in_A = Input(shape=input_shape_A, name="Input_DEM_Continuous")
    in_B = Input(shape=input_shape_B, name="Input_Radar_Params")


    # filters_list = [64, 128, 256, 512]
    filters_list = [32, 64, 128, 256]
    num_heads_list = [2, 4, 8, 16]
    repeats_list = [2, 2, 4, 2]  
    window_size = 8


    left_x = layers.Conv2D(filters_list[0], kernel_size=2, strides=2, padding='valid', name="Linear_Embedding")(in_A)


    mid_x = layers.Conv2D(filters_list[0], kernel_size=3, strides=2, padding='same', name="Mid_Stem_Conv")(in_B)

    skip_connections = []  # 

    # --------------------------------------------------
    #  Encoder: Stage 1 to Stage 4
    # --------------------------------------------------
    for i in range(4):
        stage_filters = filters_list[i]

        if i > 0:
    
            left_x = layers.Conv2D(stage_filters, kernel_size=3, strides=2, padding='same')(left_x)
            mid_x = layers.Conv2D(stage_filters, kernel_size=3, strides=2, padding='same')(mid_x)

        # ---AFIM---
        num_AFE = repeats_list[i] if i != 2 else 4
        for j in range(num_AFE):  # Stage 3 的 AFEM 标了 x4
            left_x = AFIM(left_x, stage_filters, num_heads_list[i], window_size,
                                k=kernel_size, name_prefix=f"Stage{i + 1}_AFIM_{j + 1}")

        # --- Residual blocks + SCFRM ---
        for j in range(repeats_list[i]):
            mid_x = ResidualBlock(mid_x, stage_filters)
        mid_x = SCFRM(mid_x, stage_filters, name_prefix=f"Stage{i + 1}_SCE")

        # ---  DGCFM ---
        fused_feat = DGCFM(left_x, mid_x, stage_filters, name_prefix=f"Stage{i + 1}_DGCF")

        skip_connections.append(fused_feat)
        
        mid_x = fused_feat
       
    skip_connections = skip_connections[::-1]

    dec_x = skip_connections[0]


    for i in range(1, 4):
        decoder_filters = filters_list[4 - i - 1]  # 256 -> 128 -> 64

        dec_x = layers.Conv2DTranspose(decoder_filters, kernel_size=3, strides=2, padding='same', name=f"TConv_{i}")(
            dec_x)

        dec_x = layers.Concatenate(name=f"Concat_{i}")([dec_x, skip_connections[i]])

        dec_x = layers.Conv2D(decoder_filters, 3, padding='same')(dec_x)
        dec_x = layers.BatchNormalization()(dec_x)
        dec_x = layers.Activation('relu')(dec_x)

    dec_x = layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', name="Final_TConv")(dec_x)
    dec_x = layers.Conv2D(1, 3, padding='same')(dec_x)
    dec_x = layers.BatchNormalization()(dec_x)
    dec_x = layers.Activation('relu')(dec_x)

    final_output = layers.Conv2D(1, kernel_size=1, activation='linear', name="Radar_Clutter_Map")(dec_x)

    model = models.Model(inputs=[in_A, in_B], outputs=final_output, name="DualBranch_Transformer_Model")

    model.summary()

    return model


if __name__ == "__main__":
    model = OverallNetwork(input_shape_A=(128, 128, 1), input_shape_B=(128, 128, 2))


