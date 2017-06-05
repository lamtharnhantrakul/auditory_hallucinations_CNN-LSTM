import h5py
import numpy as np

# This piece of code tests the ability to create a h5py file, append data to it, close it, then open
# it again...

USE_TITANX = False

video_1 = np.random.rand(4,224,224,3)
video_2 = np.random.rand(10,224,224,3)
video_3 = np.random.rand(23,224,224,3)
audio_vector_1 = np.random.rand(4,18)
audio_vector_2 = np.random.rand(10,18)
audio_vector_3 = np.random.rand(23,18)
audio_vector_4 = np.random.rand(25,18)
audio_vector_5 = np.random.rand(26,18)
audio_vector_6 = np.random.rand(27,18)

videos = [video_1, video_2, video_3]

video_4 = np.random.rand(25,224,224,3)
video_5 = np.random.rand(26,224,224,3)
video_6 = np.random.rand(27,224,224,3)
videos_2 = [video_4, video_5, video_6]

audio_vectors = [audio_vector_1, audio_vector_2, audio_vector_3, audio_vector_4, audio_vector_5, audio_vector_6]

if USE_TITANX:
    data_extern_dest = '/home/zanoi/ZANOI/auditory_hallucinations_data/TopAngle_data/'
else:  # Working on MacBook Pro
    data_extern_dest = '/Volumes/SAMSUNG_SSD_256GB/ADV_CV/data/'
file_name = data_extern_dest + 'TopAngleFinal_dataX_dataY.h5'

num_videos = 3
''''
print ("Creating h5py file")
dataset = h5py.File(file_name, 'w')
dataset.create_dataset('video1', data=video_1)
dataset.create_dataset('video2', data=video_2)
dataset.create_dataset('video3', data=video_3)
dataset.close()
'''


print ("Creating h5py file")
for idx, video_data in enumerate(videos):
    name = 'video' + str(idx+1)
    with h5py.File(file_name, 'a') as hf:
        print("Writing data to file...")
        print(name)
        hf.create_dataset(name, data=video_data)

print ("writing to first set of data complete")

print ('writing second set of data')

for idx, video_data in enumerate(videos_2):
    name = 'video' + str(idx+4)
    with h5py.File(file_name, 'a') as hf:
        print("Writing data to file...")
        print(name)
        hf.create_dataset(name, data=video_data)

print("writing to second set of data complete")

print ("writing audio vectors to h5py file")
for idx, audio_data in enumerate(audio_vectors):
    name = 'audio' + str(idx+1)
    with h5py.File(file_name, 'a') as hf:
        print("Writing data to file...")
        print(name)
        hf.create_dataset(name, data=audio_data)

# Open the h5py file
with h5py.File(file_name, 'r') as data:
    print("Reading data from file..")
    for key in data.keys():
        print(data[key].name)
        print(data[key].shape)

