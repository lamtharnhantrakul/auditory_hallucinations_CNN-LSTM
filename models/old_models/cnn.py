'''

import os
import numpy as np
np.random.seed(1987) #for reproducibility
import hickle as hkl
from skimage import transform, color, io
import scipy.io as sio
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

def load_data(img_path,labels_path):
    path, dirs, files = os.walk("labels_path").next()
    number_videos = len(files)

    for vid_idx in range(number_videos)
        # Read in audio data to labels
        mat_contents = sio.loadmat("../seq4_audio_vectors.mat")  # 18 x n-2
        audio_vectors = mat_contents['audio_vectors']
        print audio_vectors.shape
        Y_vid = np.matrix.transpose(audio_vectors)
        print Y_vid.shape # should be previous shape but flipped indices

        for file_idx in range(files)
        # Read in image data
        for i in range(num_images-2)
            img =
            img = skimage.transform.resize(img, (224, 224), preserve_range=True) #resize images to 224 x 224
            img *= 255.0/img.max() #mean of image

            X_rgb_vid() = img
            space_time_img = np.concatenate(color.rgb2gray(img_before), color.rgb2gray(img), color.rgb2gray(img_after))
            X_space_time_vid() = space_time_img

            img_before = img
            img = img_after
            img_after = io.imread('')
        Y = np.concatenate(Y, Y_vid)
        X_rgb = np.concatenate(X_rgb, X_rgb_vid)
        X_space_time = np.concatenate(X_space_time, X_space_time_vid)


    # Save to hickle file in ./hickle_dataset
    hkl.dump(X_rgb, './hickle_dataset/X_rgb.hkl', mode='w')
    hkl.dump(X_space_time, './hickle_dataset/X_space_time.hkl', mode='w')
    hkl.dump(Y, './hickle_dataset/Y.hkl', mode='w')

    return X_rgb, X_space_time, Y

def alex_net(weights)
    model = Sequential()
    model.add(Convolution2D(64, 3, 11, 11, border_mode='full'))
    model.add(BatchNormalization((64,226,226)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Convolution2D(128, 64, 7, 7, border_mode='full'))
    model.add(BatchNormalization((128,115,115)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Convolution2D(192, 128, 3, 3, border_mode='full'))
    model.add(BatchNormalization((128,112,112)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Convolution2D(256, 192, 3, 3, border_mode='full'))
    model.add(BatchNormalization((128,108,108)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(3, 3)))

    model.add(Flatten())
    model.add(Dense(12*12*256, 4096, init='normal'))
    model.add(BatchNormalization(4096))
    model.add(Activation('relu'))
    model.add(Dense(4096, 4096, init='normal'))
    model.add(BatchNormalization(4096))
    model.add(Activation('relu'))
    model.add(Dense(4096, 1000, init='normal'))
    model.add(BatchNormalization(1000))
    model.add(Activation('softmax'))
    return model

# Loading data
if need_load == 1
    X_rgb, X_space_time, Y = load_data('~/marimba_data','./audio_vector')
else
    X_rgb = hkl.load('./hkl_dataset/X_rgb.hkl')
    X_space_time = hkl.load('./hkl_dataset/X_space_time.hkl')
    Y = hkl.load('./hkl_dataset/Y.hkl')

merged = Merge([alex_net(weights='imagenet',rgb_img,labels), alex_net(weights='imagenet',space_time,labels)], mode='concat')
final_model = Sequential()
final_model.add(merged)
# final_model.add(Dense(18, activation='softmax'))
# hist = final_model.fit(X, y, validation_split=0.2,shuffle=True)
# print(hist.history)
fc7_map = final_model.predict(X_rgb, X_space_time)

'''