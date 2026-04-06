# -*- coding: utf-8 -*-
"""
Train an 8-step autoregressive Hybrid CNN-ConvLSTM model on packed data.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from collections import OrderedDict
from glob import glob
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv3D,
    ConvLSTM2D,
    Dropout,
    Input,
    Reshape,
    SpatialDropout3D,
)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_global_seed(seed: int) -> None:
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        LOGGER.warning("Deterministic ops are not fully supported in this environment.")


def configure_tensorflow(enable_mixed_precision: bool) -> None:
    """Configure TensorFlow runtime."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("No GPU was detected.")

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            LOGGER.warning("Failed to enable memory growth: %s", exc)

    if enable_mixed_precision:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        LOGGER.info("Mixed precision enabled.")

    logical_gpus = tf.config.list_logical_devices("GPU")
    LOGGER.info("Visible physical GPUs: %d", len(gpus))
    LOGGER.info("Visible logical GPUs: %d", len(logical_gpus))


def ensure_dir(path: str) -> None:
    """Create directory if needed."""
    os.makedirs(path, exist_ok=True)


def find_pack_dirs(base_dir: str) -> List[str]:
    """Find packed event directories."""
    pack_dirs = sorted(
        path for path in glob(os.path.join(base_dir, "*")) if os.path.isdir(path)
    )
    if not pack_dirs:
        raise FileNotFoundError(f"No packed event directories found under: {base_dir}")
    return pack_dirs


def save_json(data: Dict[str, Any], output_path: str) -> None:
    """Save a dictionary as JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_history(history: keras.callbacks.History, output_path: str) -> None:
    """Save training history to JSON."""
    history_dict = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    save_json(history_dict, output_path)


def build_runtime_paths(config: Dict[str, Any]) -> Dict[str, str]:
    """Build output file paths."""
    run_dir = config["RUN_DIR"]
    ensure_dir(run_dir)

    return {
        "best_backbone_weights": os.path.join(run_dir, "best_backbone.weights.h5"),
        "last_backbone_weights": os.path.join(run_dir, "last_backbone.weights.h5"),
        "history_json": os.path.join(run_dir, "history.json"),
        "train_log_csv": os.path.join(run_dir, "train_log.csv"),
        "config_json": os.path.join(run_dir, "train_config.json"),
    }


def save_train_config(
    config: Dict[str, Any],
    runtime_paths: Dict[str, str],
    train_dirs: Sequence[str],
    val_dirs: Sequence[str],
) -> None:
    """Save training configuration."""
    config_to_save = dict(config)
    config_to_save["n_train_events"] = len(train_dirs)
    config_to_save["n_val_events"] = len(val_dirs)
    config_to_save["train_events"] = [os.path.basename(path) for path in train_dirs]
    config_to_save["val_events"] = [os.path.basename(path) for path in val_dirs]
    save_json(config_to_save, runtime_paths["config_json"])


class ARPackSequence(keras.utils.Sequence):
    """Keras sequence for packed autoregressive rollout training."""

    def __init__(
        self,
        pack_dirs: Sequence[str],
        batch_size: int,
        rollout_steps: int,
        input_t: int,
        patch_h: int,
        patch_w: int,
        dyn_channels: int,
        static_channels: int,
        shuffle: bool = True,
        cache_size: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pack_dirs = list(pack_dirs)
        self.batch_size = int(batch_size)
        self.rollout_steps = int(rollout_steps)
        self.input_t = int(input_t)
        self.patch_h = int(patch_h)
        self.patch_w = int(patch_w)
        self.dyn_channels = int(dyn_channels)
        self.static_channels = int(static_channels)
        self.shuffle = bool(shuffle)
        self.cache_size = int(cache_size)

        self.sample_index: List[Tuple[int, int]] = []
        self.event_meta: List[Dict[str, Any]] = []
        self.cache: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()

        self._build_index()
        self.on_epoch_end()

    def _build_index(self) -> None:
        """Build a global sample index from meta.json files."""
        self.sample_index = []
        self.event_meta = []

        for event_idx, pack_dir in enumerate(self.pack_dirs):
            meta_path = os.path.join(pack_dir, "meta.json")
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            self.event_meta.append(meta)
            ar_samples = int(meta["ar_samples"])

            for ar_idx in range(ar_samples):
                self.sample_index.append((event_idx, ar_idx))

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return math.ceil(len(self.sample_index) / self.batch_size)

    def on_epoch_end(self) -> None:
        """Shuffle samples after each epoch if enabled."""
        if self.shuffle:
            random.shuffle(self.sample_index)

    def _load_event(self, event_idx: int) -> Dict[str, Any]:
        """Load one event with a small LRU cache."""
        if event_idx in self.cache:
            data = self.cache.pop(event_idx)
            self.cache[event_idx] = data
            return data

        pack_dir = self.pack_dirs[event_idx]
        with open(os.path.join(pack_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)

        data = {
            "X": np.load(os.path.join(pack_dir, "X_scaled.npy"), mmap_mode="r"),
            "Y": np.load(os.path.join(pack_dir, "Y_scaled.npy"), mmap_mode="r"),
            "Zp": np.load(os.path.join(pack_dir, "Z_patch_scaled.npy"), mmap_mode="r"),
            "meta": meta,
        }

        self.cache[event_idx] = data
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

        return data

    def __getitem__(self, batch_idx: int) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Return one batch.

        x_batch: (B, rollout_steps, input_t, H, W, Cd)
        z_batch: (B, H, W, Cs)
        y_batch: (B, rollout_steps, H, W, 1)
        """
        batch_pairs = self.sample_index[
            batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
        ]
        batch_size = len(batch_pairs)

        x_batch = np.empty(
            (
                batch_size,
                self.rollout_steps,
                self.input_t,
                self.patch_h,
                self.patch_w,
                self.dyn_channels,
            ),
            dtype=np.float32,
        )
        z_batch = np.empty(
            (batch_size, self.patch_h, self.patch_w, self.static_channels),
            dtype=np.float32,
        )
        y_batch = np.empty(
            (batch_size, self.rollout_steps, self.patch_h, self.patch_w, 1),
            dtype=np.float32,
        )

        for i, (event_idx, ar_idx) in enumerate(batch_pairs):
            data = self._load_event(event_idx)
            x_all = data["X"]
            y_all = data["Y"]
            z_all = data["Zp"]
            meta = data["meta"]

            patches_per_frame = int(meta["patches_per_frame"])
            ar_starts = int(meta["ar_starts_per_patch"])
            windows_per_patch = x_all.shape[0] // patches_per_frame

            patch_idx = ar_idx // ar_starts
            start_t = ar_idx % ar_starts

            if start_t + self.rollout_steps > windows_per_patch:
                raise ValueError(
                    f"Rollout index out of range: start_t={start_t}, "
                    f"rollout_steps={self.rollout_steps}, "
                    f"windows_per_patch={windows_per_patch}."
                )

            ids = [
                patch_idx + (start_t + step_idx) * patches_per_frame
                for step_idx in range(self.rollout_steps)
            ]

            x_batch[i] = np.asarray(x_all[ids], dtype=np.float32)
            y_batch[i] = np.asarray(y_all[ids], dtype=np.float32)
            z_batch[i] = np.asarray(z_all[patch_idx], dtype=np.float32)

        return (x_batch, z_batch), y_batch


def create_hybrid_model(
    input_time_steps: int,
    patch_h: int,
    patch_w: int,
    dyn_channels: int,
    static_channels: int,
    dynamic_filters: int,
    static_filters: int,
    dropout_rate: float,
    use_bn: bool,
    name: str = "Hybrid_Model",
) -> Model:
    """Build the Hybrid CNN-ConvLSTM backbone."""
    dynamic_input = Input(
        shape=(input_time_steps, patch_h, patch_w, dyn_channels),
        name="Dynamic_Input",
    )
    static_input = Input(
        shape=(patch_h, patch_w, static_channels),
        name="Static_Input",
    )

    dynamic_feat = ConvLSTM2D(
        dynamic_filters,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=False,
        activation="relu",
        name="Dyn_Enc_ConvLSTM",
    )(dynamic_input)
    if use_bn:
        dynamic_feat = BatchNormalization(name="Dyn_Enc_BN")(dynamic_feat)

    dynamic_feat = Reshape(
        (1, patch_h, patch_w, dynamic_filters),
        name="Dyn_Enc_ReshapeToT1",
    )(dynamic_feat)
    dynamic_feat = ConvLSTM2D(
        dynamic_filters,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="relu",
        name="Dyn_Dec_ConvLSTM",
    )(dynamic_feat)
    dynamic_feat = SpatialDropout3D(dropout_rate, name="Dyn_SpDrop3D")(dynamic_feat)

    static_feat = Conv2D(
        static_filters,
        kernel_size=(5, 5),
        activation="relu",
        padding="same",
        name="Sta_Conv2D_5",
    )(static_input)
    static_feat = Conv2D(
        static_filters,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="Sta_Conv2D_3",
    )(static_feat)
    static_feat = Conv2D(
        static_filters,
        kernel_size=(1, 1),
        activation="relu",
        padding="same",
        name="Sta_Conv2D_1",
    )(static_feat)
    static_feat = Dropout(dropout_rate, name="Sta_Drop")(static_feat)
    static_feat = Reshape(
        (1, patch_h, patch_w, static_filters),
        name="Sta_ReshapeToT1",
    )(static_feat)

    fused_feat = Concatenate(axis=-1, name="Fuse_Concat")([dynamic_feat, static_feat])
    fused_feat = Conv3D(
        dynamic_filters,
        kernel_size=(1, 1, 1),
        padding="same",
        activation="relu",
        name="Fuse_1x1x1",
    )(fused_feat)

    output = Conv3D(
        1,
        kernel_size=(3, 7, 7),
        padding="same",
        activation="linear",
        name="Out_Conv3D",
    )(fused_feat)

    return Model(inputs=[dynamic_input, static_input], outputs=output, name=name)


def weighted_mse_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Weighted MSE with higher weight on wet pixels."""
    weights = tf.where(y_true > 0.001, 1.0, 0.1)
    weights = tf.where(y_true <= 0.0, 0.0, weights)
    return tf.reduce_mean(weights * tf.square(y_true - y_pred))


def huber_mean(y_true: tf.Tensor, y_pred: tf.Tensor, delta: float = 0.1) -> tf.Tensor:
    """Mean Huber loss."""
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    per_pixel = 0.5 * tf.square(quadratic) + delta * linear
    return tf.reduce_mean(per_pixel)


def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Equal-weight combination of weighted MSE and Huber loss."""
    return 0.5 * weighted_mse_loss(y_true, y_pred) + 0.5 * huber_mean(
        y_true, y_pred, delta=0.1
    )


def roll_window_with_pred_depth(
    x_window: tf.Tensor,
    next_frame_exogenous: tf.Tensor,
    pred_depth: tf.Tensor,
) -> tf.Tensor:
    """Update the rolling input window with predicted depth."""
    pred_depth_2d = pred_depth[:, 0, :, :, 0]

    new_last = tf.identity(next_frame_exogenous)
    new_last = tf.concat([pred_depth_2d[..., None], new_last[..., 1:]], axis=-1)
    new_last = new_last[:, None, ...]

    return tf.concat([x_window[:, 1:], new_last], axis=1)


class ARHybridTrainer(Model):
    """Autoregressive trainer wrapper for multi-step rollout."""

    def __init__(
        self,
        backbone: Model,
        rollout_steps: int,
        future_weights: Sequence[float],
        loss_fn=combined_loss,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.rollout_steps = int(rollout_steps)
        self.future_weights = tf.constant(list(future_weights), dtype=tf.float32)
        self.teacher_forcing_ratio = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        self.loss_fn = loss_fn

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.reg_tracker = keras.metrics.Mean(name="reg_loss")
        self.roll_mae_tracker = keras.metrics.Mean(name="roll_mae")

        self.val_loss_tracker = keras.metrics.Mean(name="val_loss")
        self.val_reg_tracker = keras.metrics.Mean(name="val_reg_loss")
        self.val_roll_mae_tracker = keras.metrics.Mean(name="val_roll_mae")

    @property
    def metrics(self) -> List[keras.metrics.Metric]:
        """Metrics reset automatically by Keras."""
        return [
            self.loss_tracker,
            self.reg_tracker,
            self.roll_mae_tracker,
            self.val_loss_tracker,
            self.val_reg_tracker,
            self.val_roll_mae_tracker,
        ]

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: bool = False) -> tf.Tensor:
        """Accept either a single window or a rollout tensor."""
        x_input, z = inputs
        x_rank = x_input.shape.rank

        if x_rank == 6:
            x_window = x_input[:, 0]
        elif x_rank == 5:
            x_window = x_input
        else:
            raise ValueError(f"Unexpected input rank: {x_input.shape}")

        return self.backbone([x_window, z], training=training)

    def _rollout(
        self,
        x_steps: tf.Tensor,
        z: tf.Tensor,
        y_future: tf.Tensor | None = None,
        training: bool = False,
        use_teacher_forcing: bool = False,
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """Run multi-step rollout."""
        x_window = x_steps[:, 0]
        preds: List[tf.Tensor] = []
        step_loss_list: List[tf.Tensor] = []

        for step_idx in range(self.rollout_steps):
            y_pred_step = self.backbone([x_window, z], training=training)
            preds.append(y_pred_step)

            if y_future is not None:
                y_true_step = y_future[:, step_idx : step_idx + 1]
                step_loss_list.append(self.loss_fn(y_true_step, y_pred_step))

            if step_idx < self.rollout_steps - 1:
                next_frame_exogenous = x_steps[:, step_idx + 1, -1]

                if use_teacher_forcing and (y_future is not None):
                    y_true_step = y_future[:, step_idx : step_idx + 1]
                    rand_mask = tf.less(
                        tf.random.uniform(
                            (tf.shape(x_steps)[0], 1, 1, 1, 1),
                            minval=0.0,
                            maxval=1.0,
                        ),
                        self.teacher_forcing_ratio,
                    )
                    depth_to_feed = tf.where(rand_mask, y_true_step, y_pred_step)
                else:
                    depth_to_feed = y_pred_step

                x_window = roll_window_with_pred_depth(
                    x_window=x_window,
                    next_frame_exogenous=next_frame_exogenous,
                    pred_depth=depth_to_feed,
                )

        preds = tf.concat(preds, axis=1)
        return preds, step_loss_list

    def train_step(self, data: Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Custom training step."""
        (x_steps, z), y_future = data

        with tf.GradientTape() as tape:
            preds, step_loss_list = self._rollout(
                x_steps=x_steps,
                z=z,
                y_future=y_future,
                training=True,
                use_teacher_forcing=True,
            )

            step_losses = tf.stack(step_loss_list, axis=0)
            supervised_loss = (
                tf.reduce_sum(step_losses * self.future_weights)
                / tf.reduce_sum(self.future_weights)
            )
            regularization_loss = (
                tf.add_n(self.backbone.losses) if self.backbone.losses else 0.0
            )
            total_loss = supervised_loss + regularization_loss

        grads = tape.gradient(total_loss, self.backbone.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.backbone.trainable_variables))

        rollout_mae = tf.reduce_mean(tf.abs(y_future - preds))
        self.loss_tracker.update_state(total_loss)
        self.reg_tracker.update_state(regularization_loss)
        self.roll_mae_tracker.update_state(rollout_mae)

        logs = {
            "loss": self.loss_tracker.result(),
            "reg_loss": self.reg_tracker.result(),
            "roll_mae": self.roll_mae_tracker.result(),
            "tf_ratio": self.teacher_forcing_ratio,
        }
        for i in range(self.rollout_steps):
            logs[f"step{i + 1}_loss"] = step_losses[i]
        return logs

    def test_step(self, data: Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Custom validation step."""
        (x_steps, z), y_future = data

        preds, step_loss_list = self._rollout(
            x_steps=x_steps,
            z=z,
            y_future=y_future,
            training=False,
            use_teacher_forcing=False,
        )

        step_losses = tf.stack(step_loss_list, axis=0)
        supervised_loss = (
            tf.reduce_sum(step_losses * self.future_weights)
            / tf.reduce_sum(self.future_weights)
        )
        regularization_loss = tf.add_n(self.backbone.losses) if self.backbone.losses else 0.0
        total_loss = supervised_loss + regularization_loss

        rollout_mae = tf.reduce_mean(tf.abs(y_future - preds))
        self.val_loss_tracker.update_state(total_loss)
        self.val_reg_tracker.update_state(regularization_loss)
        self.val_roll_mae_tracker.update_state(rollout_mae)

        logs = {
            "loss": self.val_loss_tracker.result(),
            "reg_loss": self.val_reg_tracker.result(),
            "roll_mae": self.val_roll_mae_tracker.result(),
        }
        for i in range(self.rollout_steps):
            logs[f"step{i + 1}_loss"] = step_losses[i]
        return logs


class TeacherForcingScheduler(tf.keras.callbacks.Callback):
    """Linear teacher-forcing schedule."""

    def __init__(
        self,
        warmup_epochs: int,
        decay_start: int,
        decay_end: int,
        final_ratio: float,
    ) -> None:
        super().__init__()
        self.warmup_epochs = int(warmup_epochs)
        self.decay_start = int(decay_start)
        self.decay_end = int(decay_end)
        self.final_ratio = float(final_ratio)

    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any] | None = None) -> None:
        """Update teacher-forcing ratio at epoch start."""
        if epoch < self.warmup_epochs:
            ratio = 1.0
        elif epoch < self.decay_end:
            span = max(1, self.decay_end - self.decay_start)
            alpha = (epoch - self.decay_start) / span
            alpha = min(max(alpha, 0.0), 1.0)
            ratio = 1.0 * (1.0 - alpha) + self.final_ratio * alpha
        else:
            ratio = self.final_ratio

        self.model.teacher_forcing_ratio.assign(ratio)
        LOGGER.info("Epoch %d: teacher_forcing_ratio = %.4f", epoch + 1, ratio)


class SaveBackboneWeights(tf.keras.callbacks.Callback):
    """Save backbone weights when the monitored metric improves."""

    def __init__(self, save_path: str, monitor: str = "val_loss", mode: str = "min") -> None:
        super().__init__()
        self.save_path = save_path
        self.monitor = monitor
        self.mode = mode
        self.best = np.inf if mode == "min" else -np.inf

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any] | None = None) -> None:
        """Save best backbone weights."""
        logs = logs or {}
        value = logs.get(self.monitor)
        if value is None:
            return

        improved = (value < self.best) if self.mode == "min" else (value > self.best)
        if improved:
            self.best = value
            self.model.backbone.save_weights(self.save_path)
            LOGGER.info(
                "Epoch %d: %s = %.6f, saved to %s",
                epoch + 1,
                self.monitor,
                float(value),
                self.save_path,
            )


def main() -> None:
    """Main entry point."""
    config: Dict[str, Any] = {
        "PACK_ROOT": "./data/hybrid_ar_rollout_packed_8step",
        "PACK_TRAIN_DIR": "./data/hybrid_ar_rollout_packed_8step/train",
        "PACK_VAL_DIR": "./data/hybrid_ar_rollout_packed_8step/val",
        "RUN_DIR": "./outputs/hybrid_ar_rollout_train_packed_8step",
        "SEED": 42,
        "USE_BN": False,
        "USE_MIXED_PRECISION": False,
        "INPUT_T": 4,
        "ROLLOUT_STEPS": 8,
        "TOTAL_T": 8,
        "PATCH_H": 100,
        "PATCH_W": 100,
        "DYN_CHANNELS": 4,
        "STATIC_CHANNELS": 12,
        "EPOCHS": 60,
        "BATCH_SIZE": 64,
        "INIT_LR": 1e-3,
        "CLIPNORM": 5.0,
        "DYNAMIC_FILTERS": 64,
        "STATIC_FILTERS": 64,
        "DROPOUT_RATE": 0.1,
        "SHUFFLE_TRAIN": True,
        "TRAIN_CACHE_SIZE": 8,
        "VAL_CACHE_SIZE": 2,
        "TF_WARMUP_EPOCHS": 5,
        "TF_DECAY_START": 5,
        "TF_DECAY_END": 40,
        "TF_FINAL_RATIO": 0.2,
        "FUTURE_WEIGHTS": [1.0] * 8,
    }

    if config["TOTAL_T"] != config["ROLLOUT_STEPS"]:
        raise ValueError(
            f"Expected TOTAL_T == ROLLOUT_STEPS, got TOTAL_T={config['TOTAL_T']} "
            f"and ROLLOUT_STEPS={config['ROLLOUT_STEPS']}."
        )

    runtime_paths = build_runtime_paths(config)

    set_global_seed(config["SEED"])
    configure_tensorflow(config["USE_MIXED_PRECISION"])

    strategy = tf.distribute.MirroredStrategy()
    LOGGER.info("MirroredStrategy replicas: %d", strategy.num_replicas_in_sync)

    train_dirs = find_pack_dirs(config["PACK_TRAIN_DIR"])
    val_dirs = find_pack_dirs(config["PACK_VAL_DIR"])
    LOGGER.info("Train events: %d", len(train_dirs))
    LOGGER.info("Validation events: %d", len(val_dirs))

    save_train_config(
        config=config,
        runtime_paths=runtime_paths,
        train_dirs=train_dirs,
        val_dirs=val_dirs,
    )

    train_seq = ARPackSequence(
        pack_dirs=train_dirs,
        batch_size=config["BATCH_SIZE"],
        rollout_steps=config["ROLLOUT_STEPS"],
        input_t=config["INPUT_T"],
        patch_h=config["PATCH_H"],
        patch_w=config["PATCH_W"],
        dyn_channels=config["DYN_CHANNELS"],
        static_channels=config["STATIC_CHANNELS"],
        shuffle=config["SHUFFLE_TRAIN"],
        cache_size=config["TRAIN_CACHE_SIZE"],
    )
    val_seq = ARPackSequence(
        pack_dirs=val_dirs,
        batch_size=config["BATCH_SIZE"],
        rollout_steps=config["ROLLOUT_STEPS"],
        input_t=config["INPUT_T"],
        patch_h=config["PATCH_H"],
        patch_w=config["PATCH_W"],
        dyn_channels=config["DYN_CHANNELS"],
        static_channels=config["STATIC_CHANNELS"],
        shuffle=False,
        cache_size=config["VAL_CACHE_SIZE"],
    )

    with strategy.scope():
        backbone = create_hybrid_model(
            input_time_steps=config["INPUT_T"],
            patch_h=config["PATCH_H"],
            patch_w=config["PATCH_W"],
            dyn_channels=config["DYN_CHANNELS"],
            static_channels=config["STATIC_CHANNELS"],
            dynamic_filters=config["DYNAMIC_FILTERS"],
            static_filters=config["STATIC_FILTERS"],
            dropout_rate=config["DROPOUT_RATE"],
            use_bn=config["USE_BN"],
            name="Hybrid_Model_ARTrain_Packed_8Step",
        )

        model = ARHybridTrainer(
            backbone=backbone,
            rollout_steps=config["ROLLOUT_STEPS"],
            future_weights=config["FUTURE_WEIGHTS"],
            loss_fn=combined_loss,
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config["INIT_LR"],
            clipnorm=config["CLIPNORM"],
        )
        model.compile(optimizer=optimizer, run_eagerly=False)

    callbacks = [
        TeacherForcingScheduler(
            warmup_epochs=config["TF_WARMUP_EPOCHS"],
            decay_start=config["TF_DECAY_START"],
            decay_end=config["TF_DECAY_END"],
            final_ratio=config["TF_FINAL_RATIO"],
        ),
        SaveBackboneWeights(
            save_path=runtime_paths["best_backbone_weights"],
            monitor="val_loss",
            mode="min",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=False,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(runtime_paths["train_log_csv"]),
    ]

    LOGGER.info("Start training.")
    LOGGER.info("Train batches: %d", len(train_seq))
    LOGGER.info("Validation batches: %d", len(val_seq))
    LOGGER.info("Batch size: %d", config["BATCH_SIZE"])

    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=config["EPOCHS"],
        callbacks=callbacks,
        verbose=1,
    )

    save_history(history, runtime_paths["history_json"])
    backbone.save_weights(runtime_paths["last_backbone_weights"])

    LOGGER.info("Training finished.")
    LOGGER.info("Best backbone weights: %s", runtime_paths["best_backbone_weights"])
    LOGGER.info("Last backbone weights: %s", runtime_paths["last_backbone_weights"])
    LOGGER.info("History JSON: %s", runtime_paths["history_json"])
    LOGGER.info("CSV log: %s", runtime_paths["train_log_csv"])


if __name__ == "__main__":
    setup_logging()
    main()
