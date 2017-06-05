import numpy as np
from keras_pretrained_models.imagenet_utils import preprocess_input
from keras.models import Model
from keras.preprocessing import image

from keras_pretrained_models.vgg19 import VGG19

base_model = VGG19(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)

img_path = '../test_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
print (x.shape)
print (np.max(x))
x = np.expand_dims(x, axis=0)
print (x.shape)
x = preprocess_input(x)
print (x.shape)
print (np.max(x))

fc2_features = model.predict(x)
print (fc2_features.shape)
for i in range(4096):
    print (fc2_features[:,i])