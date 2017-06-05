import os
from keras.models import save_model
from keras.initializers import RandomNormal
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.utils.io_utils import HDF5Matrix
import h5py
from models.my_callbacks import *
import time
from keras.utils.io_utils import HDF5Matrix
def CNN_LSTM_model(image_dim=(224,224,3), audio_vector_dim=18, learning_rate=0.4e-6, weight_init=0.01):
    (img_rows, img_cols, img_channels) = image_dim  # (224,224,3)
    input_img = Input(shape=(img_rows, img_cols, img_channels))

    # Like Hanoi's work with DeepMind Reinforcement Learning, build a model that does not use pooling layers
    # to retain sensitivty to locations of objects
    # Tried (64,128,256,512)

    x = Conv2D(filters=32,
               kernel_size=(16, 16),
               input_shape=image_dim,
               name='Input_Layer',
               activation='relu',
               padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=weight_init, seed=None),
               strides=(8, 8))(input_img)

    x = Conv2D(filters=64,
               kernel_size=(8, 8),
               activation='relu',
               padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=weight_init, seed=None),
               strides=(4, 4))(x)

    x = Conv2D(filters=128,
               kernel_size=(4, 4),
               activation='relu',
               padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=weight_init, seed=None),
               strides=(2, 2))(x)

    x = Conv2D(filters=256,
               kernel_size=(2, 2),
               input_shape=image_dim,
               activation='relu',
               padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=weight_init, seed=None),
               strides=(1, 1))(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu', name='1st_FC', kernel_initializer=RandomNormal(mean=0.0, stddev=weight_init, seed=None))(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu', name='2nd_FC', kernel_initializer=RandomNormal(mean=0.0, stddev=weight_init, seed=None))(x)
    # x = TimeDistributedDense(1)(x)

    # Note that LSTM expects input shape: (nb_samples, timesteps, feature_dim)
    x = Reshape((1, 512))(x)
    x = LSTM(256, input_shape=(1, 512), dropout=0.2, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(128, dropout=0.2, name='LSTM_reg_output')(x)
    network_output = Dense(audio_vector_dim)(x)

    model = Model(inputs=input_img, outputs=network_output)

    # Use the Adam optimizer for gradient descent
    adam = Adam(lr=learning_rate)

    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())

    return model

def CNN_LSTM_STATEFUL_model(image_dim, audio_vector_dim):
    (img_rows, img_cols, img_channels) = image_dim  # (224,224,3)
    input_img = Input(shape=(img_rows, img_cols, img_channels))

    # Like Hanoi's work with DeepMind Reinforcement Learning, build a model that does not use pooling layers
    # to retain sensitivty to locations of objects
    # Tried (64,128,256,512)

    x = Conv2D(filters=32,
               kernel_size=(16, 16),
               input_shape=image_dim,
               name='Input_Layer',
               activation='relu',
               padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
               strides=(8, 8))(input_img)

    x = Conv2D(filters=64,
               kernel_size=(8, 8),
               activation='relu',
               padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
               strides=(4, 4))(x)

    x = Conv2D(filters=128,
               kernel_size=(4, 4),
               activation='relu',
               padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
               strides=(2, 2))(x)

    x = Conv2D(filters=256,
               kernel_size=(2, 2),
               input_shape=image_dim,
               activation='relu',
               padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
               strides=(1, 1))(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu', name='1st_FC')(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu', name='2nd_FC')(x)
    # x = TimeDistributedDense(1)(x)

    # Note that LSTM expects input shape: (nb_samples, timesteps, feature_dim)
    x = Reshape((1, 512))(x)
    x = LSTM(256, input_shape=(1, 512), dropout=0.2, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(128, dropout=0.2, name='LSTM_reg_output')(x)
    network_output = Dense(audio_vector_dim)(x)

    model = Model(inputs=input_img, outputs=network_output)

    # Use the Adam optimizer for gradient descent
    adam = Adam(lr=0.4e-6)

    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())

    return model