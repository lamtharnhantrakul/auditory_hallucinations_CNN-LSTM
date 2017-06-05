### NOTE NOTE NOTE!!!

# to use this example, you need to change keras backend to theano AND also revert back to an older environment shimon_herokeras, change model


from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from skimage.transform import resize
from skimage.io import imread
from skimage import img_as_ubyte
import numpy as np
import numpy as np

import skimage as skimage

from keras.layers import Input

from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.models import save_model
from keras.optimizers import Adam

from skimage import transform, color, exposure

def VGG_16():
    input_img = Input(shape=(3,224,224))
    x = ZeroPadding2D((1,1),input_shape=(3,224,224))(input_img)
    x = Convolution2D(64, 3, 3, activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(64, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(128, 3, 3, activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(128, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256, 3, 3, activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256, 3, 3, activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(256, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu')(x)
    x = ZeroPadding2D((1,1))(x)
    x = Convolution2D(512, 3, 3, activation='relu')(x)
    x = MaxPooling2D((2,2), strides=(2,2))(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    aux_output = x
    prob_out = Dense(1000, activation='softmax')(x)

    model = Model(input=input_img, output=[prob_out, aux_output])
    model.load_weights('vgg16_weights.h5')

    return model

if __name__ == "__main__":
    im = np.random.rand(224,224,3)
    #im[:,:,0] -= 103.939
    #im[:,:,1] -= 116.779
    #im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    print (im.shape)
    im = np.expand_dims(im, axis=0)

    # Test pretrained model
    model = VGG_16()

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    model_weights = model.get_weights()
    for layer in model_weights:
        print (layer.shape)
    print (model_weights[1])
    #out = model.predict(im)
    #print (np.argmax(out))