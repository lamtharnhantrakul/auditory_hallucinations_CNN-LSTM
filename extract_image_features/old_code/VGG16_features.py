import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

# dimensions of our images.
img_width, img_height = 224, 224

top_model_weights_path = 'bottleneck_fc_model.h5'


def save_bottlebeck_features(video_array):
    # build the VGG16 network
    model = applications.VGG16(include_top=True, weights='imagenet')

    # video_array should be (None,224,224,1)
    bottleneck_features = model.predict(video_array)
    return bottleneck_features

'''
def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array(
        [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)
'''

video_array = np.random.rand(2,224,224,3) # Remember that we concatenate 3 grey scale images into 1 image and treat that like "RGB"
print (video_array.shape)
bottleneck_features = save_bottlebeck_features(video_array=video_array)
print ("bottleneck_features.shape:", bottleneck_features.shape)

# We have 18 audio vectors
#train_top_model()