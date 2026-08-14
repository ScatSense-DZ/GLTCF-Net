import math
import tensorflow as tf
from tensorflow import keras


class WeightedPixelSSIMLoss(keras.layers.Layer):
    """
    L_total = alpha * L_pixel + beta * L_ssim

    alpha = sigmoid(raw_alpha)
    beta  = 1 - alpha

    因此始终满足 alpha + beta = 1。
    """

    def __init__(
        self,
        initial_alpha=0.5,
        min_weight=0.0,
        max_val=1.0,
        filter_size=11,
        filter_sigma=1.5,
        k1=0.01,
        k2=0.03,
        name="weighted_pixel_ssim_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        if not 0.0 < initial_alpha < 1.0:
            raise ValueError(
                "initial_alpha 必须严格位于 (0, 1) 内"
            )

        if not 0.0 <= min_weight < 0.5:
            raise ValueError(
                "min_weight 必须位于 [0, 0.5) 内"
            )

        if max_val <= 0:
            raise ValueError("max_val 必须大于 0")

        if not (
            min_weight
            < initial_alpha
            < 1.0 - min_weight
        ):
            raise ValueError(
                "initial_alpha 必须位于 "
                f"({min_weight}, {1.0 - min_weight}) 内"
            )

        self.initial_alpha = float(initial_alpha)
        self.min_weight = float(min_weight)
        self.max_val = float(max_val)
        self.filter_size = int(filter_size)
        self.filter_sigma = float(filter_sigma)
        self.k1 = float(k1)
        self.k2 = float(k2)

        # 如果 min_weight > 0：
        #
        # alpha = min_weight
        #       + (1 - 2 * min_weight) * sigmoid(raw_alpha)
        #
        # 先根据 initial_alpha 反推出 raw_alpha。
        alpha_ratio = (
            initial_alpha - min_weight
        ) / (
            1.0 - 2.0 * min_weight
        )

        initial_raw_alpha = math.log(
            alpha_ratio / (1.0 - alpha_ratio)
        )

        self.raw_alpha = self.add_weight(
            name="raw_alpha",
            shape=(),
            initializer=keras.initializers.Constant(
                initial_raw_alpha
            ),
            trainable=True,
            dtype=tf.float32
        )

    @property
    def alpha(self):
        alpha_ratio = tf.sigmoid(self.raw_alpha)

        return (
            self.min_weight
            + (1.0 - 2.0 * self.min_weight)
            * alpha_ratio
        )

    @property
    def beta(self):
        # 不单独计算 sigmoid，直接用 1-alpha，
        # 从参数化上保证二者之和为 1。
        return 1.0 - self.alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # 像素损失。这里使用 MAE。
        pixel_loss = tf.reduce_mean(
            tf.abs(y_true - y_pred)
        )

        # SSIM 输入应和 max_val 对应。
        y_true_ssim = tf.clip_by_value(
            y_true,
            0.0,
            self.max_val
        )
        y_pred_ssim = tf.clip_by_value(
            y_pred,
            0.0,
            self.max_val
        )

        ssim = tf.image.ssim(
            y_true_ssim,
            y_pred_ssim,
            max_val=self.max_val,
            filter_size=self.filter_size,
            filter_sigma=self.filter_sigma,
            k1=self.k1,
            k2=self.k2
        )

        mean_ssim = tf.reduce_mean(ssim)
        mean_ssim = tf.clip_by_value(
            mean_ssim,
            -1.0,
            1.0
        )

        # mean_ssim 位于 [-1, 1]，
        # 因此 ssim_loss 位于 [0, 2]。
        ssim_loss = 1.0 - mean_ssim

        current_alpha = self.alpha
        current_beta = 1.0 - current_alpha

        weighted_pixel_loss = (
            current_alpha * pixel_loss
        )
        weighted_ssim_loss = (
            current_beta * ssim_loss
        )

        total_loss = (
            weighted_pixel_loss
            + weighted_ssim_loss
        )

        # 定位 NaN、Inf 或意外的负值。
        tf.debugging.assert_all_finite(
            total_loss,
            "total_loss 出现 NaN 或 Inf"
        )
        tf.debugging.assert_non_negative(
            pixel_loss,
            "pixel_loss 出现负数"
        )
        tf.debugging.assert_non_negative(
            ssim_loss,
            "ssim_loss 出现负数"
        )
        tf.debugging.assert_non_negative(
            total_loss,
            "total_loss 出现负数"
        )
        tf.debugging.assert_near(
            current_alpha + current_beta,
            tf.constant(1.0, dtype=tf.float32),
            atol=1e-6,
            message="alpha + beta 不等于 1"
        )

        return {
            "loss": total_loss,
            "pixel_loss": pixel_loss,
            "ssim_loss": ssim_loss,
            "weighted_pixel_loss": weighted_pixel_loss,
            "weighted_ssim_loss": weighted_ssim_loss,
            "alpha": current_alpha,
            "beta": current_beta,
            "mean_ssim": mean_ssim
        }

    def get_config(self):
        config = super().get_config()

        config.update({
            "initial_alpha": self.initial_alpha,
            "min_weight": self.min_weight,
            "max_val": self.max_val,
            "filter_size": self.filter_size,
            "filter_sigma": self.filter_sigma,
            "k1": self.k1,
            "k2": self.k2
        })

        return config


class WeightedLossModel(keras.Model):
    def __init__(
        self,
        base_model,
        loss_layer,
        name="weighted_loss_model",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.base_model = base_model
        self.loss_layer = loss_layer

        self.loss_tracker = keras.metrics.Mean(
            name="loss"
        )
        self.pixel_loss_tracker = keras.metrics.Mean(
            name="pixel_loss"
        )
        self.ssim_loss_tracker = keras.metrics.Mean(
            name="ssim_loss"
        )
        self.weighted_pixel_loss_tracker = keras.metrics.Mean(
            name="weighted_pixel_loss"
        )
        self.weighted_ssim_loss_tracker = keras.metrics.Mean(
            name="weighted_ssim_loss"
        )
        self.alpha_tracker = keras.metrics.Mean(
            name="alpha"
        )
        self.beta_tracker = keras.metrics.Mean(
            name="beta"
        )


    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.pixel_loss_tracker,
            self.ssim_loss_tracker,
            self.weighted_pixel_loss_tracker,
            self.weighted_ssim_loss_tracker,
            self.alpha_tracker,
            self.beta_tracker,
            self.regularization_loss_tracker
        ]

    def call(self, inputs, training=False):
        return self.base_model(
            inputs,
            training=training
        )

    def calculate_losses(self, x, y, training=False):
        y_pred = self(
            x,
            training=training
        )

        values = self.loss_layer(
            y,
            y_pred
        )

        # 常规 L1/L2 正则项是非负的。
        if self.base_model.losses:
            regularization_loss = tf.add_n([
                tf.cast(value, tf.float32)
                for value in self.base_model.losses
            ])
        else:
            regularization_loss = tf.constant(
                0.0,
                dtype=tf.float32
            )

        tf.debugging.assert_non_negative(
            regularization_loss,
            "模型中存在负的正则化损失"
        )

        total_loss = (
            values["loss"]
            + regularization_loss
        )

        tf.debugging.assert_all_finite(
            total_loss,
            "total_loss 出现 NaN 或 Inf"
        )
        tf.debugging.assert_non_negative(
            total_loss,
            "total_loss 出现负数"
        )

        return values, regularization_loss, total_loss

    def update_loss_metrics(
        self,
        values,
        regularization_loss,
        total_loss
    ):
        self.loss_tracker.update_state(
            total_loss
        )
        self.pixel_loss_tracker.update_state(
            values["pixel_loss"]
        )
        self.ssim_loss_tracker.update_state(
            values["ssim_loss"]
        )
        self.weighted_pixel_loss_tracker.update_state(
            values["weighted_pixel_loss"]
        )
        self.weighted_ssim_loss_tracker.update_state(
            values["weighted_ssim_loss"]
        )
        self.alpha_tracker.update_state(
            values["alpha"]
        )
        self.beta_tracker.update_state(
            values["beta"]
        )
        self.regularization_loss_tracker.update_state(
            regularization_loss
        )

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            (
                values,
                regularization_loss,
                total_loss
            ) = self.calculate_losses(
                x,
                y,
                training=True
            )

        gradients = tape.gradient(
            total_loss,
            self.trainable_variables
        )

        gradients_and_variables = [
            (gradient, variable)
            for gradient, variable in zip(
                gradients,
                self.trainable_variables
            )
            if gradient is not None
        ]

        self.optimizer.apply_gradients(
            gradients_and_variables
        )

        self.update_loss_metrics(
            values,
            regularization_loss,
            total_loss
        )

        return {
            metric.name: metric.result()
            for metric in self.metrics
        }

    def test_step(self, data):
        x, y = data

        (
            values,
            regularization_loss,
            total_loss
        ) = self.calculate_losses(
            x,
            y,
            training=False
        )

        self.update_loss_metrics(
            values,
            regularization_loss,
            total_loss
        )

        return {
            metric.name: metric.result()
            for metric in self.metrics
        }
class PrintLossCallback(keras.callbacks.Callback):
    def __init__(self, print_every=1):
        super().__init__()
        self.print_every = int(print_every)

    @staticmethod
    def format_value(logs, key):
        value = logs.get(key)

        if value is None:
            return "N/A"

        if hasattr(value, "numpy"):
            value = value.numpy()

        return f"{float(value):.6f}"

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if (epoch + 1) % self.print_every != 0:
            return

        keys = [
            "loss",
            "alpha",
            "beta",
            "pixel_loss",
            "ssim_loss",
            "weighted_pixel_loss",
            "weighted_ssim_loss",
            "val_loss",
            "val_pixel_loss",
            "val_ssim_loss",
            "val_weighted_pixel_loss",
            "val_weighted_ssim_loss",
        ]

        text = ", ".join(
            f"{key}={self.format_value(logs, key)}"
            for key in keys
        )

        print(
            f"\nEpoch {epoch + 1:05d}: {text}"
        )
