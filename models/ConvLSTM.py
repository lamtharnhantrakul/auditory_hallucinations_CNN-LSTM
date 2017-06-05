import scipy.io as sio
import os
import numpy as np
from collections import deque
import pickle
from keras.models import save_model
from keras.initializers import RandomNormal
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras import metrics
import h5py
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization

USE_TITANX = True


def createModel(input_sequence_dim, audio_vector_dim):
    frame_h, frame_w, channels = input_sequence_dim  # (80,80,1) we are going to pass in single greyscale images

    # ConvLSTM expects an input with shape
    # (n_frames, width, height, channels)  In the example they used n_frames = 15
    input_sequence = Input(shape=(None, frame_h, frame_w, channels))


    # input to a ConvLSTM2D is of form (samples, time, filters, output_row, output_col)`
    x = ConvLSTM2D(filters=32,
                   kernel_size=(8, 8),
                   padding='same',
                   strides=(4, 4),
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                   return_sequences=True)(input_sequence)
    x = BatchNormalization()(x)

    x = ConvLSTM2D(filters=64,
                   kernel_size=(4, 4),
                   padding='same',
                   strides=(2, 2),
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                   return_sequences=True)(x)
    x = BatchNormalization()(x)

    x = ConvLSTM2D(filters=64,
                   kernel_size=(3, 3),
                   padding='same',
                   strides=(1, 1),
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                   return_sequences=True)(x)
    x = BatchNormalization()(x)

    '''
    x = ConvLSTM2D(filters=64,
                   kernel_size=(2, 2),
                   padding='same',
                   strides=(1, 1),
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
                   return_sequences=True)(x)
    x = BatchNormalization()(x)
    '''

    # The Conv3D layer causes the the hidden "volumes" to become a flat image
    # The 3D convolution changes a (None, None, 40, 40, 40) into a (None,None,40,40,40)
    # Output of a Conv3D is (samples, filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)
    x = Conv3D(filters=1,
               kernel_size=(3, 3, 3),
               activation='sigmoid',
               padding='same',
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
               data_format='channels_last')(x)

    '''
    # Run a CNN over the "flat" image
    x = Conv2D(filters=32,
               kernel_size=(3, 3),
               strides=(1, 1),
               kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=None),
               padding='same',
               activation='relu')(x)
    '''

    x = Flatten()(x)
    # x = Dense(1024, activation='relu', name='1st_FC')(x)
    # x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    network_output = Dense(audio_vector_dim, name='regression_out')(x)

    model = Model(inputs=input_sequence, outputs=network_output)

    # Use the Adam optimizer for gradient descent
    adam = Adam(lr=0.5e-6)
    model.compile(loss='mse', optimizer=adam)

createModel((80,80,1),18)
