
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

# 24/4/2017
# This is a CNN+LSTM intended to train from scratch directly from the video frames with dimensions 200x200

USE_TITANX = True

def createModel(image_dim, audio_vector_dim):
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


# Testing if the model compiles
#model = createModel((224, 224, 3), 18)
print(">>> STARTING TIME:", str(time.strftime("%m-%d_%H-%M-%S")))

#############
### READING THE DATASET
# Define the external SSD where the dataset residesin
if USE_TITANX:
    data_dir = '/home/zanoi/ZANOI/auditory_hallucinations_data/'
else:
    data_dir = '/Volumes/SAMSUNG_SSD_256GB/ADV_CV/data/'

data_file = data_dir + 'TopAngle100_dataX_dataY.h5'

# Load first element of data to extract information on video
with h5py.File(data_file, 'r') as hf:
    print("Reading data from file..")
    dataX_sample = hf['dataX_train'][0]
    dataY_sample = hf['dataY_train'][0]
    print("dataX_sample.shape:", dataX_sample.shape)
    print("dataY_sample.shape:", dataY_sample.shape)

    dataX_train = hf['dataX_train']  # Adding the [:] actually loads it into memory
    dataY_train = hf['dataY_train']
    dataX_test = hf['dataX_test']
    dataY_test = hf['dataY_test']
    print("dataX_train.shape:", dataX_train.shape)
    print("dataY_train.shape:", dataY_train.shape)
    print("dataX_test.shape:", dataX_test.shape)
    print("dataY_test.shape:", dataY_test.shape)

(frame_h, frame_w, channels) = dataX_sample.shape  # (8377,224,224,3)
audio_vector_dim = dataY_sample.shape[0]

# Load data into HDF5Matrix object, which reads the file from disk and does not put it into RAM
dataX_train = HDF5Matrix(data_file, 'dataX_train')
dataY_train = HDF5Matrix(data_file, 'dataY_train')
dataX_test = HDF5Matrix(data_file, 'dataX_test')
dataY_test = HDF5Matrix(data_file, 'dataY_test')

#############
### BUILD THE MODEL
model = createModel((frame_h, frame_w, channels), audio_vector_dim)

#############
### CHECK FOR EXISTING MODEL
#model_name = ''
#if os.path.exists("../checkpoints/" + model_name):
    #model.load_weights("../checkpoints/" + model_name)

#############
###Define a bunch of callbacks
# Set up Keras checkpoints to monitor the loss and only save when it improves
#filepath="../checkpoints/CNN_LSTM_scratch_v3-{epoch:02d}-{loss:.5f}.hdf5"
#checkpointCallBack = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

# Setup tensorboard
# tbCallBack = TensorBoard(log_dir='./graph', histogram_freq=1, write_graph=True, write_images=True)

# Uses a special callback class from models.my_callbacks
testSeqCallback = predictSeqCallback()
# Put these in a callback list
callbacks_list = [testSeqCallback]


#############
### BEGIN TRAINING THE MODEL
# This function actually starts the training
# model.fit(dataX, dataY, epochs=500, batch_size=256, callbacks=callbacks_list, verbose=2)
# The maximum batch size that fits on the TitanX Pascal is 10,000
model.fit(dataX_test, dataY_test, shuffle='batch', epochs=20, batch_size=10000, validation_data=(dataX_test, dataY_test), verbose=1, callbacks = callbacks_list)

# Model saving now occurs in the callback on epoch end

print("Saving trained model...")
model_prefix = 'CNN_LSTM_scratch_v3'
model_path = "../trained_models/" + model_prefix + ".h5"
save_model(model, model_path, overwrite=True)  # saves weights, network topology and optimizer state (if any)


print(">>> ENDING TIME:", str(time.strftime("%m-%d_%H-%M-%S")))
print("--- {EVERYTHING COMPLETE HOMIEEEEEEEEE} ---")
