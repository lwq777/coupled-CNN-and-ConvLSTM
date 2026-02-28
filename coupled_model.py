import datetime
import pickle
from threading import Lock

import numpy as np
import os
from random import shuffle
from glob import glob
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential, Model
from keras.layers import (Conv3D, ReLU, Reshape, Conv2D, Concatenate, Dropout, LayerNormalization,
                          ConvLSTM2D, MaxPooling2D, UpSampling2D, Input, Lambda)
from keras.layers import BatchNormalization,Multiply
from keras.losses import Huber
from keras.optimizers import Adam
from keras.saving.save import load_model
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams

# Set font for matplotlib
rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei font
rcParams['axes.unicode_minus'] = False  # Ensure minus signs are displayed correctly

matplotlib.use("Agg")


# Custom scaler class: Incremental Robust Scaler
class IncrementalRobustScaler(BaseEstimator, TransformerMixin):
    def __init__(self, sample_limit=10000):
        self.sample_limit = sample_limit
        self.samples_ = None
        self.center_ = None
        self.scale_ = None

    def partial_fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.samples_ is None:
            self.samples_ = X.copy()
        else:
            self.samples_ = np.vstack([self.samples_, X])
            if len(self.samples_) > self.sample_limit:
                idx = np.random.choice(len(self.samples_), self.sample_limit, replace=False)
                self.samples_ = self.samples_[idx]
        self.center_ = np.median(self.samples_, axis=0)
        q1 = np.percentile(self.samples_, 25, axis=0)
        q3 = np.percentile(self.samples_, 75, axis=0)
        self.scale_ = q3 - q1
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def fit(self, X, y=None):
        return self.partial_fit(X, y)

    def transform(self, X):
        X = np.asarray(X)
        if self.center_ is None or self.scale_ is None:
            raise ValueError("Must call fit or partial_fit before transform.")
        return (X - self.center_) / self.scale_

    def inverse_transform(self, X_scaled):
        X_scaled = np.asarray(X_scaled)
        if self.center_ is None or self.scale_ is None:
            raise ValueError("Must call fit or partial_fit before inverse_transform.")
        return X_scaled * self.scale_ + self.center_

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

def get_vali_samples(station_dirs):
    Ys = []
    Xs = []
    Zs = []
    count = 0
    for station_dir in station_dirs:

        y_path = os.path.join(station_dir, 'Y.npy')
        x_path = os.path.join(station_dir, 'X.npy')
        z_path = os.path.join(station_dir, 'Z.npy')

        Y = np.load(y_path)  # (samples, 4, 100, 100, 1)
        X = np.load(x_path)  # (samples, 1, 100, 100, 4)
        Z = np.load(z_path)  # (samples, 100, 100, 10)
        Ys.append(Y)
        Xs.append(X)
        Zs.append(Z)
        count += 1
        if count == 2:
            break
    return np.concatenate(Xs, axis=0), np.concatenate(Zs, axis=0), np.concatenate(Ys, axis=0)

# Function to fit the scalers on the data
def fit_scaler(station_dirs):
    """
    Fit MinMaxScaler for dynamic inputs, outputs, and static features.

    :param station_dirs: List of station directory paths.
    :return: Dictionary containing scalers for dynamic inputs, outputs, and static features.
    """
    dynamic_scalers = [MinMaxScaler(feature_range=(0, 1)) for _ in range(4)]  # 4 dynamic input channels
    output_scaler = MinMaxScaler(feature_range=(0, 1))  # Single dynamic output channel
    static_scalers = [MinMaxScaler(feature_range=(0, 1)) for _ in range(12)]  # 12 static feature channels

    for station_dir in tqdm(station_dirs):
        x_path = os.path.join(station_dir, 'X.npy')
        y_path = os.path.join(station_dir, 'Y.npy')
        z_path = os.path.join(station_dir, 'Z.npy')

        X = np.load(x_path, mmap_mode='r')
        Y = np.load(y_path, mmap_mode='r+')
        Z = np.load(z_path, mmap_mode='r')

        # Process dynamic input channels (4 channels)
        for channel in range(4):
            channel_data = X[..., channel]
            valid_data = channel_data[channel_data != -9999].reshape(-1, 1)
            if valid_data.size > 0:
                dynamic_scalers[channel].partial_fit(valid_data)

        # Process dynamic output
        Y[Y < 0] = 0
        valid_data = Y[Y != -9999].reshape(-1, 1)
        if valid_data.size > 0:
            output_scaler.partial_fit(valid_data)

        # Process static features
        for channel in range(Z.shape[-1]):
            channel_data = Z[..., channel]
            valid_data = channel_data[channel_data != -9999].reshape(-1, 1)
            if valid_data.size > 0:
                static_scalers[channel].partial_fit(valid_data)

    return {
        "dynamic_scalers": dynamic_scalers,
        "output_scaler": output_scaler,
        "static_scalers": static_scalers
    }


# Data generator function
def smart_data_generator(station_dirs, batch_size, scalers, buffer_size=5, verbose=True):
    """
    Generates data batches for training with dynamic inputs, outputs, and static features.

    :param station_dirs: List of station directory paths.
    :param batch_size: Batch size.
    :param scalers: Dictionary containing MinMaxScalers for dynamic inputs, outputs, and static features.
    :param buffer_size: Number of stations to buffer in memory.
    :param verbose: Whether to print debug info.
    :return: A generator yielding data batches.
    """
    dynamic_scalers = scalers["dynamic_scalers"]
    output_scaler = scalers["output_scaler"]
    static_scalers = scalers["static_scalers"]
    lock = Lock()

    pool = []  # Cache pool [(X, Y, Z, name)]
    station_indices = list(range(len(station_dirs)))
    current_station_ptr = 0  # Pointer to traverse all stations

    def load_station_data(station_dir):
        station_name = os.path.basename(station_dir)
        if verbose:
            print(f"[Loading] Station: {station_name}")
        X = np.load(os.path.join(station_dir, 'X.npy'))
        Y = np.load(os.path.join(station_dir, 'Y.npy'))
        Z = np.load(os.path.join(station_dir, 'Z.npy'))

        for ch in range(4):
            X[..., ch] = dynamic_scalers[ch].transform(X[..., ch].reshape(-1, 1)).reshape(X[..., ch].shape)

        Y[Y == -9999] = 0
        Y = output_scaler.transform(Y.reshape(-1, 1)).reshape(Y.shape)

        Z_mask = (Z != -9999).astype(np.float32)
        Z_mask = np.expand_dims(np.max(Z_mask, axis=-1), axis=-1)
        Z[Z == -9999] = 0
        for ch in range(Z.shape[-1]):
            Z[..., ch] = static_scalers[ch].transform(Z[..., ch].reshape(-1, 1)).reshape(Z[..., ch].shape)
        Z = np.concatenate([Z, Z_mask], axis=-1)

        return X, Y, Z, station_name

    # Initialize buffer with first `buffer_size` stations
    for _ in range(buffer_size):
        station_dir = station_dirs[station_indices[current_station_ptr % len(station_dirs)]]
        pool.append(load_station_data(station_dir))
        current_station_ptr += 1

    pool_idx = 0  # Index of current pool data
    sample_offset = 0

    while True:
        # Get current station data from the pool
        X_block, Y_block, Z_block, station_name = pool[pool_idx]
        num_samples = X_block.shape[0]

        if sample_offset + batch_size <= num_samples:
            idx = np.arange(sample_offset, sample_offset + batch_size)
            sample_offset += batch_size
        else:
            # If current station data is exhausted, load new station
            station_dir = station_dirs[station_indices[current_station_ptr % len(station_dirs)]]
            pool[pool_idx] = load_station_data(station_dir)
            current_station_ptr += 1
            sample_offset = 0
            pool_idx = (pool_idx + 1) % buffer_size
            continue

        batch_X = X_block[idx]
        batch_Y = Y_block[idx]
        batch_Z = Z_block[idx]

        if verbose:
            print(f"[Generating] Station: {station_name} | Sample Index: {sample_offset - batch_size}~{sample_offset}")

        with lock:
            yield [batch_X, batch_Z], batch_Y


# Model creation function
def create_model():
    input_time_steps = 4
    w, h, c = 100, 100, 4
    static_channels = 12
    dynamic_filters = 64
    static_filters = 64
    output_time_steps = 1
    dropout_rate = 0.2

    # Dynamic input
    encoder_input = Input(shape=(input_time_steps, w, h, c), name="Dynamic_Input")

    # Single ConvLSTM encoder layer
    encoder_output = ConvLSTM2D(filters=dynamic_filters, kernel_size=(5, 5), padding='same',
                                return_sequences=False, name="Encoder_ConvLSTM")(encoder_input)
    encoder_output = BatchNormalization()(encoder_output)
    encoder_output = Dropout(dropout_rate)(encoder_output)

    # Decoder input (repeat time dimension)
    decoder_input = Reshape((1, w, h, dynamic_filters))(encoder_output)
    decoder_input = Lambda(lambda x: tf.repeat(x, repeats=output_time_steps, axis=1))(decoder_input)

    # Decoder ConvLSTM
    decoder_output = ConvLSTM2D(filters=dynamic_filters, kernel_size=(5, 5), padding='same',
                                return_sequences=True, name="Decoder_ConvLSTM")(decoder_input)
    decoder_output = Dropout(dropout_rate)(decoder_output)

    # Static input
    static_input = Input(shape=(w, h, static_channels + 1), name="Static_Input")
    static_features = static_input[..., :-1]
    static_mask = static_input[..., -1:]

    static_features = Conv2D(static_filters, (5, 5), activation='relu', padding='same')(static_features)
    static_features = Conv2D(static_filters, (3, 3), activation='relu', padding='same')(static_features)
    static_features = Conv2D(static_filters, (1, 1), activation='relu', padding='same')(static_features)
    static_features = BatchNormalization()(static_features)
    static_features = Dropout(dropout_rate)(static_features)
    static_features = Multiply()([static_features, static_mask])
    static_features = Reshape((1, w, h, static_filters))(static_features)
    static_features = tf.tile(static_features, [1, output_time_steps, 1, 1, 1])

    # Fusion of dynamic + static
    fused = Concatenate(axis=-1)([decoder_output, static_features])

    # Output layer
    output = Conv3D(
        filters=1,
        kernel_size=(3, 3, 3),
        activation='linear',
        padding='same',
        name="Final_Output_Conv3D"
    )(fused)

    model = Model(inputs=[encoder_input, static_input], outputs=output, name="ConvLSTM_1Layer_NoSkip")
    model.summary()
    return model


# Loss functions and training setup
def weighted_mse_loss(y_true, y_pred):
    weights = tf.where(y_true > 0.01, 1.0, 0.1)
    loss = tf.reduce_mean(weights * tf.square(y_true - y_pred))
    return loss


def nse_loss(y_true, y_pred):
    num = tf.reduce_sum(tf.square(y_true - y_pred))
    den = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - num / (den + 1e-6)  # Avoid division by zero


def combined_loss(y_true, y_pred):
    return 0.6 * weighted_mse_loss(y_true, y_pred) + 0.4 * Huber(delta=0.1)(y_true, y_pred)


# Main training script
if __name__ == "__main__":
    base_dir = 'F:/datasets_with_9geo_1threshold_0.2overlap_masked/process'
    station_dirs = sorted(glob(os.path.join(base_dir, '*')))
    shuffle(station_dirs)

    # Split into training and validation sets
    num_stations = len(station_dirs)
    train_stations = station_dirs[:int(0.8 * num_stations)]
    val_stations = station_dirs[int(0.8 * num_stations):]

    true_x, true_z, true_y = get_vali_samples(val_stations)

    # Fit scalers on the data
    scalers = fit_scaler(station_dirs)

    # Save scalers for later use
    with open('scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)

    # Load scalers
    scalers = pickle.load(open('scalers.pkl', 'rb'))

    # Create data generators
    batch_size = 32
    train_gen = smart_data_generator(train_stations, batch_size, scalers, buffer_size=4)
    val_gen = smart_data_generator(val_stations, batch_size, scalers, buffer_size=1)

    # Model setup
    model = create_model()
    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss=combined_loss, metrics=['mae'])

    # Callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0000001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=8)

    # Model training
    model.fit(
        train_gen,
        steps_per_epoch=len(train_stations) // batch_size,
        validation_data=val_gen,
        validation_steps=len(val_stations) // batch_size,
        callbacks=[reduce_lr, early_stopping],
        epochs=50
    )

    now = datetime.datetime.now().strftime("%Y-%m-%d")
    model.save(f'model_merge-{now}-2.h5')

    # Load and test the model
    model = load_model(f'model_merge-{now}-2.h5', custom_objects={'combined_loss': combined_loss})
    pre_y = model.predict(val_gen, steps=len(val_stations) // batch_size)
    print(pre_y.shape)
