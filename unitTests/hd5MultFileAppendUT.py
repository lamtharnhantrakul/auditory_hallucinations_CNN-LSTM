import h5py
import numpy as np

# This piece of code tests the ability to create a h5py file, append data to it, close it, then open
# it again...



USE_TITANX = False

video_1 = np.random.rand(4,224,224,3)
video_2 = np.random.rand(10,224,224,3)
video_3 = np.random.rand(23,224,224,3)
video_4 = np.random.rand(25,224,224,3)
video_5 = np.random.rand(26,224,224,3)
video_6 = np.random.rand(27,224,224,3)

audio_vector_1 = np.random.rand(4,18)
audio_vector_2 = np.random.rand(10,18)
audio_vector_3 = np.random.rand(23,18)
audio_vector_4 = np.random.rand(25,18)
audio_vector_5 = np.random.rand(26,18)
audio_vector_6 = np.random.rand(27,18)

videos = [video_1, video_2, video_3, video_4, video_5, video_6]
audio_vectors = [audio_vector_1, audio_vector_2, audio_vector_3, audio_vector_4, audio_vector_5, audio_vector_6]

if USE_TITANX:
    data_extern_dest = '/home/zanoi/ZANOI/auditory_hallucinations_data/TopAngle_data/'
else:  # Working on MacBook Pro
    data_extern_dest = '/Volumes/SAMSUNG_SSD_256GB/ADV_CV/data/'
file_name = data_extern_dest + 'TopAngleFinal_dataX_dataY.h5'

'''
print ("writing first video")
with h5py.File(file_name, 'w') as f:
    dset = f.create_dataset("dataX", (4,224,224,3),maxshape=(None,244,244,3))
    dset[:] = video_1
    print(dset.shape)


with h5py.File(file_name, 'a') as hf:
    dset = hf['dataX']
    next_video_numFrames = video_2.shape[0]
    dset.resize(dset.len() + next_video_numFrames, axis=0)
    dset[-next_video_numFrames:] = video_2
    print(dset.shape)


with h5py.File(file_name, 'a') as hf:
    dset = hf['dataX']
    next_video_numFrames = video_3.shape[0]
    dset.resize(dset.len() + next_video_numFrames, axis=0)
    dset[-next_video_numFrames:] = video_3
    print(dset.shape)
'''

for i in range(len(videos)):
    if i == 0:
        # If this is the first video file, you need to create the first matrix
        with h5py.File(file_name, 'w') as f:
            video_dset = f.create_dataset("dataX", videos[0].shape, maxshape=(None, 244, 244, 3))
            video_dset[:] = videos[0]
            print("video_dset.shape:", video_dset.shape)

            audio_dset = f.create_dataset("dataY", audio_vectors[0].shape, maxshape=(None,18))
            audio_dset[:] = audio_vectors[0]
            print("audio_dset.shape:", audio_dset.shape)
    else: # Then simply append to the last matrix
        with h5py.File(file_name, 'a') as hf:
            video_dset = hf['dataX']
            current_video = videos[i]
            numFrames = current_video.shape[0]
            video_dset.resize(video_dset.len() + numFrames, axis=0)
            video_dset[-numFrames:] = current_video
            print("video_dset.shape:", video_dset.shape)


            audio_dset = hf['dataY']
            current_audioVec = audio_vectors[i]
            numFrames = current_audioVec.shape[0]
            audio_dset.resize(audio_dset.len() + numFrames, axis=0)
            audio_dset[-numFrames:] = current_audioVec
            print("audio_dset.shape:", audio_dset.shape)

'''
print ("Creating h5py file")
for idx, video_data in enumerate(videos):
    name = 'video' + str(idx+1)
    with h5py.File(file_name, 'a') as hf:
        print("Writing data to file...")
        print(name)
        hf.create_dataset(name, data=video_data)

print ("writing video_data complete)

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

'''

