import scipy.io as sio
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import pylab as plt

LSTM_LOOK_BACK = 1  # Number of time steps to look back

### { PREPARE THE AUDIO DATA } ###

# Read the audio_vectors generated from MATLAB
audio_vectors_dir = './audio_vectors'
files = [os.path.join(audio_vectors_dir, fname)
        for fname in os.listdir(audio_vectors_dir)
        if fname.endswith('.mat')]

# Concatenate the audio vectors from each video into one long matrix
audio_dataset = np.array([])
for file_name in files:
    mat_contents = sio.loadmat(file_name)
    curr_audio_vector = mat_contents['audio_vectors']
    audio_dataset = np.hstack([audio_dataset, curr_audio_vector]) if audio_dataset.size else curr_audio_vector
audio_dataset = audio_dataset.T  # In MATLAB, plotting the audio_vectors is easy if matrix is of form (18, n). For training, we must use (n,18)
#print audio_dataset.shape

# convert an array of values into a dataset matrix
def create_dataset(dataset, LSTM_LOOK_BACK=3):
    data = []
    for i in range(len(dataset) - LSTM_LOOK_BACK - 1):
        a = dataset[i:(i + LSTM_LOOK_BACK), 0]
        dataX.append(a)
    return np.array(data)

Y = audio_dataset
dataY = Y[0:2089, :]  # Select only the first video
print dataY.shape

# As we wait for FC7 features to process, for now just use a dummy variable
X = np.random.rand(len(audio_dataset), 4096*2)  # There is a 4096 from the 3 channel grayscale stacked images, and another 4096 from the RGB image.
dataX = X[0:2089, :]
print dataX.shape

# reshape X to be [samples, time steps, features]
dataX = np.reshape(dataX, (dataX.shape[0], LSTM_LOOK_BACK, -1))
print dataX.shape

### { DEFINE THE CONV_LSTM MODEL } ####
model = Sequential()
model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   input_shape=(None, 40, 40, 1),
                   padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
model.add(BatchNormalization())

model.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
               activation='sigmoid',
               padding='same', data_format='channels_last'))
# output shape = (None,None,40,40,1)

model.add(Dense(Y.shape[1]))

model.compile(loss='mean_squared_error', optimizer='adam')
print (model.summary())

# define the checkpoint
filepath = "./checkpoints/LSTM-weights-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Begin training process
model.fit(dataX, dataY, nb_epoch=20, batch_size=256, verbose=2, callbacks=callbacks_list)
