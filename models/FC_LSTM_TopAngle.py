import os
from keras.models import save_model
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Dense, Dropout
from keras.layers import LSTM
import h5py
import time
from keras.utils.io_utils import HDF5Matrix
from models.my_callbacks import *

USE_TITANX = True

def createModel(input_sequence_dim, audio_vector_dim):
    # input_sequence_dim: tuple of dimensions e.g (1,4096)
    # audio_vector_dim: int of dimension e.g 18
    timesteps, features = input_sequence_dim
    input_sequences = Input(shape=(timesteps, features))  # (1,4096)
    # Note that LSTM expects input shape: (nb_samples, timesteps, feature_dim)

    x = LSTM(256, dropout=0.2, return_sequences=True, name='LSTM_layer1')(input_sequences)
    x = Dropout(0.2)(x)
    x = LSTM(256, dropout=0.2, name='LSTM_layer2')(x)
    network_output = Dense(audio_vector_dim, name='regression_out')(x)

    model = Model(inputs=input_sequences, outputs=network_output)

    # Use the Adam optimizer for gradient descent
    adam = Adam(lr=0.5e-6)

    #model.compile(loss='mean_squared_error', validation_split=0.1, optimizer='adam')
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())

    return model

# Testing if the model compiles
# model = createModel((1,4096), 18)

print(">>> STARTING TIME:", str(time.strftime("%m-%d_%H-%M-%S")))

#############
### READING THE DATASET
# Define the external SSD where the dataset residesin
if USE_TITANX:
    data_dir = '/home/zanoi/ZANOI/auditory_hallucinations_data/'
else:
    data_dir = '/Volumes/SAMSUNG_SSD_256GB/ADV_CV/data/'

data_file = data_dir + 'TopAngleFC1_dataX_dataY.h5'

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

(frame_h, frame_w, channels) = dataX_sample.shape  # (8377,1,4096)
audio_vector_dim = dataY_sample.shape[0]

# Load data into HDF5Matrix object, which reads the file from disk and does not put it into RAM
dataX_train = HDF5Matrix(data_file, 'dataX_train')
dataY_train = HDF5Matrix(data_file, 'dataY_train')
dataX_test = HDF5Matrix(data_file, 'dataX_test')
dataY_test = HDF5Matrix(data_file, 'dataY_test')

timesteps = 1
features = 4096

#############
### BUILD THE MODEL
model = createModel((timesteps, features),audio_vector_dim)

#############

# Uses a special callback class from models.my_callbacks
testSeqCallback = predictSeqCallback()
# Put these in a callback list
callbacks_list = [testSeqCallback]

# This function actually starts the training
#model.fit(dataX, dataY, epochs=500, batch_size=256, callbacks=callbacks_list, verbose=2)
model.fit(dataX_train, dataY_train, epochs=500, batch_size=10000, validation_data=[dataX_test,dataY_test], verbose=1, callbacks=callbacks_list)

print ("Saving trained model...")
model_prefix = 'FC_LSTM_TopAngleFC1_v2'
model_path = "../trained_models/" + model_prefix + ".h5"
save_model(model, model_path, overwrite=True)  # saves weights, network topology and optimizer state (if any)

print ("--- {EVERYTHING COMPLETE HOMIEEEEEEEEE} ---")
