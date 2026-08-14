

"""
    Implementation of Callback
    @author: Zhu Dong
    @date: 2025.04.18

"""

import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class LossHistory(Callback):
    def __init__(self, model):
        super(LossHistory, self).__init__()
        self.model = model
        self.ssim_history = []
        self.mse_history = []
        self.cos_history = []
        self.weights_history = []

    def on_epoch_end(self, epoch, logs=None):
        # 获取当前 WeightedLoss 层的实例
        weighted_loss_layer = self.model.get_layer("weighted_loss")  # 确保层有name

        # 提取标量值（强制转换为 Python float）
        ssim_val = weighted_loss_layer.ssim_loss.numpy().item()  # 关键：.item()
        mse_val = weighted_loss_layer.mse_loss.numpy().item()
        cos_val = weighted_loss_layer.cos_loss.numpy().item()
        weights_val = weighted_loss_layer.lossWeights.numpy().tolist()  # 转为列表

        # 记录到日志
        logs["ssim_loss"] = ssim_val
        logs["mse_loss"] = mse_val
        logs["cos_loss"] = cos_val
        logs["lossWeights"] = weights_val

        # 保存历史记录
        self.ssim_history.append(ssim_val)
        self.mse_history.append(mse_val)
        self.cos_history.append(cos_val)
        self.weights_history.append(weights_val)
        # 打印日志
        print(
            f"\nEpoch {epoch}: "
            f"SSIM={ssim_val:.4f}, "
            f"MSE={mse_val:.4f}, "
            f"Cosine={cos_val:.4f}, "
            f"Weights={weights_val}"
        )

