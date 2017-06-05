import h5py
import numpy as np

# Join top angle movies together into one large data file

USE_TITANX = True

# Define the external SSD where the dataset residesin
if USE_TITANX:
    data_dir = '/home/zanoi/ZANOI/auditory_hallucinations_data/'
else:
    data_dir = '/Volumes/SAMSUNG_SSD_256GB/ADV_CV/data/'

top_angles = ['seq1angle5_dataX_dataY.h5',
              'seq2angle5_dataX_dataY.h5',
              'seq3angle4_dataX_dataY.h5']

data_files = [data_dir + file_i
              for file_i in top_angles]

# Open the h5py file
dataX = []
dataY = []
for data_file in data_files:
    with h5py.File(data_file, 'r') as hf:
        print("Reading data from file..")
        dataX_vid = hf['dataX'][:]
        dataY_features = hf['dataY'][:]
    print("dataX_vid.shape:", dataX_vid.shape)
    print("dataY_features.shape:", dataY_features.shape)
    dataX.append(dataX_vid)
    dataY.append(dataY_features)

# manual code, REALLY BAD!
final_dataX = np.concatenate((dataX[0],dataX[1]), axis=0)
final_dataX = np.concatenate((final_dataX,dataX[2]), axis=0)

final_dataY = np.concatenate((dataY[0],dataY[1]), axis=0)
final_dataY = np.concatenate((final_dataY,dataY[2]), axis=0)

print ("final_dataX.shape:", final_dataX.shape)
print ("final_dataY.shape:", final_dataY.shape)

file_name = data_dir + 'TopAnglesFC2_dataX_dataY.h5'
with h5py.File(file_name, 'w') as hf:
    print("Writing data to file...")
    hf.create_dataset('dataX', data=final_dataX)
    hf.create_dataset('dataY', data=final_dataY)

print ("--- {EVERYTHING COMPLETE HOMIEEEEEEEEE} ---")