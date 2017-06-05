import matplotlib
print(matplotlib.get_backend())

import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")
import matplotlib.image as mpimg
import numpy as np

img=np.random.rand(512,512,3)

imgplot = plt.imshow(img)
plt.ion()