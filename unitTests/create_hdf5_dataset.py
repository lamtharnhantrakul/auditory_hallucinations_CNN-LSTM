import numpy as np
from DatasetHelper import DatasetHelper
import h5py

dataset = DatasetHelper('D:/Guitar Amp Modelling/Training_nocab/')

x = np.asarray([])
y = np.asarray([])

h = h5py.File('data_nocab_new.h5','a')

#Write into HDF5 dataset
x = np.asarray([])
y = np.asarray([])
while True:
    b,flag = dataset.load_data()
    x = np.concatenate([x,b['x']])
    y = np.concatenate([y,b['y']])
    if flag:
        break
        
x = x.reshape((-1,1,1))
y = y.reshape((-1,1))
d_x = h.create_dataset('x',maxshape=(None,),data=x)
d_y = h.create_dataset('y',maxshape=(None,),data=y)

# while True:
    # if flag:
        # break
    # a,flag = dataset.load_data()
    # x = np.concatenate([x,a['x']])
    # y = np.concatenate([y,a['y']])
        
# d_x.resize((d_x.len()+x.shape[0],))
# d_x[d_x.len():] = x
# d_y.resize((d_y.len()+y.shape[0],))
# d_y[d_y.len():] = y

h.close()