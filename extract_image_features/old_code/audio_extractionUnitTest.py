from skimage import transform, color, io
import scipy.io as sio
import skvideo.io
import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
import re
from skimage import transform, color, io
import warnings
from tqdm import tqdm
import h5py
import pickle

###########
### LOADING VIDEOS ###
video_dir = "/Volumes/SAMSUNG_SSD_256GB/ADV_CV/2-25_VIDAUD/EXPORTS"

video_files = [os.path.join(video_dir, file_i)
         for file_i in os.listdir(video_dir)
         if file_i.endswith('.mp4')]

num_videos = len(video_files)
print("num_videos: ", num_videos)

############
### LOADING AUDIO ###
audio_feature_dir = "../audio_vectors"

audio_f_files = [os.path.join(audio_feature_dir, file_i)
                for file_i in os.listdir(audio_feature_dir)
                if file_i.endswith('.mat')]

num_audio_f = len(audio_f_files)
print("num_audio_f: ", num_audio_f)

###########
### READING AUDIO VECTORS
audio_f_file = audio_f_files[3]  # Test with just one audio feature vector, and find all the corresponding movies
mat_contents = sio.loadmat(audio_f_file)  # 18 x n-2
audio_vectors = mat_contents['audio_vectors']
audio_vector_length = audio_vectors.shape[1]
#print(audio_f_files[0])
print("audio_vectors.shape: ", audio_vectors.shape)

# Extract the file prefix using regular expressions
start = audio_f_file.find('seq')
end = audio_f_file.find("_audio", start)
audio_prefix = audio_f_file[start:end]

##########
### READING GREYSCALE VIDEO
vid_extern_dest = '/Volumes/SAMSUNG_SSD_256GB/ADV_CV/videos_BW/'
file_name = vid_extern_dest + audio_prefix + '_vid_BW.h5'

with h5py.File(file_name, 'r') as hf:
    print("Reading data from file..")
    video_data = hf['videos_BW'][:]
print("video_data.shape:", video_data.shape)

### TESTING this method

def createAudioVectorDataset(audio_vectors, dataX_shape):
    # audio_vectors: a numpy array of the audio vector (18,8378)
    # dataX_shape: shape of the space time image (1, 8377, 224, 224, 3)
    (num_videos, num_frames, frame_h, frame_w, channels) = dataX_shape
    final_audio_vectors = np.zeros((num_videos, num_frames, audio_vectors.shape[0]))  # (1, 18, 8377)
    single_audio_vector = audio_vectors[:, 0:num_frames]  # Extract the corresponding audio vector, produces (18, 8377) from (18, 8379)
    for i in range(num_videos):
        final_audio_vectors[i] = single_audio_vector.T  # Assign the audio_vector to each video angle in idx=0 , (1, 8377, 224, 224, 3). Need to transpose it here.
    return final_audio_vectors


final_audio_vectors = createAudioVectorDataset(audio_vectors, (1, 8377, 224, 224, 3))
print ("final_audio_vectors.shape:", final_audio_vectors.shape)