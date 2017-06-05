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

imageio.plugins.ffmpeg.download()
### HELPER FUNCTIONS ### -> need to move to a utils.py

######################
### FOR READING AND PROCESSING VIDEO FRAMES TO GREYSCALE
def findMatchingVideos(audio_prefix, video_filenames):
    # audio_prefix = string
    # video_filenames = [] list of filenames
    # Compares the video sequence prefix ("seq1") with audio sequence prefix ("seq1") to extract the related videos
    matching_videos = []
    for video_filename in video_filenames:
        start = video_filename.find('seq')
        end = video_filename.find("_angle", start)
        video_prefix = video_filename[start:end]
        if audio_prefix == video_prefix:
            matching_videos.append(video_filename)
    return matching_videos

def processOneVideo(audio_f_length, video_filename, output_dimensions, normalize=False):
    # audio_f_length = int (length of audio feature vector corresponding to the video)
    # video_filename = str (filename of a single video)
    (frame_h, frame_w) = output_dimensions
    vid = imageio.get_reader(video_filename, 'ffmpeg')
    #greyscale_vid = []
    greyscale_vid = np.zeros((audio_f_length + 1, frame_h, frame_w))
    for i in tqdm(range(audio_f_length + 1)):
        # apparently if I have an audio_vector of dimensions (18,8378), then the number of frames in the video is 8379
        with warnings.catch_warnings():  # Ignores the warnings about depacrated functions that don't apply to this code
            warnings.simplefilter("ignore")
            img = vid.get_data(i)
            if normalize:  # the pretrained network expects an image that is from 0 - 255
                img = img / np.max(img)  # rescale to 0 - 1 scale
            img = transform.resize(img, (frame_h, frame_w), preserve_range=True)  # resize images to 60x60
            img = color.rgb2gray(img)  # convert to greyscale
            greyscale_vid[i] = img
    #greyscale_vid = np.array(greyscale_vid)
    print('\n')
    print("Processed:", video_filename, "video shape:", greyscale_vid.shape)
    return greyscale_vid

def processVideos(audio_f_length, video_filenames):
    # audio_f_length = int (length of audio feature vector corresponding to the video)
    # video_filenames = [] list of filename strings
    processed_videos = []
    for video_filename in tqdm(video_filenames):  # iterate through each of the video file names
        processed_videos.append(processOneVideo(audio_f_length, video_filename))
    return np.array(processed_videos)

def returnAudioVectors(audio_idx, audio_f_files):
    audio_f_file = audio_f_files[audio_idx]  # Test with just one audio feature vector, and find all the corresponding movies
    mat_contents = sio.loadmat(audio_f_file)  # 18 x n-2
    audio_vectors = mat_contents['audio_vectors']
    audio_vector_length = audio_vectors.shape[1]
    # print(audio_f_files[0])
    #print("audio_vectors.shape: ", audio_vectors.shape)

    # Extract the file prefix using regular expressions
    start = audio_f_file.find('seq')
    end = audio_f_file.find("_audio", start)
    audio_prefix = audio_f_file[start:end]
    return audio_prefix, audio_vector_length, audio_vectors

def returnAngleName(video_filename):
    # video_filename: string of video filename e.g
    start = video_filename.find('angle')
    end = video_filename.find(".mp4", start)
    angle_name = video_filename[start:end]
    return angle_name

#########
### SPACE-TIME CONCATENATION
def createSpaceTimeImagesforOneVideo(video, CNN_window):
    # video: single video of BW images
    # video.shape: (8379, 224, 224)
    (num_frames, frame_h, frame_w) = video.shape
    space_time_single_vid = np.zeros((num_frames - (CNN_window-1), frame_h,frame_w,CNN_window))  # (8377, 224, 224, 3)
    for i in tqdm(range(num_frames - (CNN_window-1))):
        l_idx = i
        r_idx = l_idx + (CNN_window)
        curr_stack = video[l_idx:r_idx, :, :] # Extracts (3,224,224)

        # For some reason, np.reshape (170,170,3) to (3,170,170) causes a very weird duplication of images
        #curr_stack = np.reshape(curr_stack,(frame_h,frame_w,CNN_window))  # reshapes to (224,224,3)

        # Define out own reshape function
        new_stack = np.zeros((frame_h, frame_w, CNN_window))
        for j in range(CNN_window):
            new_stack[:, :, j] = curr_stack[j, :, :]

        space_time_single_vid[i,:,:,:] = new_stack
    return space_time_single_vid

def createSpaceTimeImages(video_data, CNN_window=3):
    # video_data: a numpy array
    # video_data.shape: (1, 8379, 224, 224)
    # CNN_window: int of how many frames to put together
    (num_videos, num_frames, frame_h, frame_w) = video_data.shape
    space_time_images = np.zeros((num_videos, num_frames - (CNN_window-1), frame_h, frame_w, CNN_window))   # (1, 8377, 224, 224, 3)
    for i in range(num_videos):
        space_time_images[i] = createSpaceTimeImagesforOneVideo(video_data[i],CNN_window)
    return space_time_images

def createAudioVectorDataset(audio_vectors, dataX_shape):
    # audio_vectors: a numpy array of the audio vector (18,8378)
    # dataX_shape: shape of the space time image (1, 8377, 224, 224, 3)
    (num_videos, num_frames, frame_h, frame_w, channels) = dataX_shape
    final_audio_vectors = np.zeros((num_videos, num_frames, audio_vectors.shape[0]))  # (1, 18, 8377)
    # Hanoi, you forgot to normalize the audio vectors in MATLAB!!!
    audio_vectors = audio_vectors/np.max(audio_vectors)  # normalize the audio vectors
    single_audio_vector = audio_vectors[:, 0:num_frames]  # Extract the corresponding audio vector, produces (18, 8377) from (18, 8379)
    for i in range(num_videos):
        final_audio_vectors[i] = single_audio_vector.T  # Assign the audio_vector to each video angle in idx=0 , (1, 8377, 224, 224, 3). Need to transpose it here.
    return final_audio_vectors

def createAudioVectorDatasetForOneVid(audio_vectors, dataX_shape):
    # audio_vectors: a numpy array of the audio vector (18,8378)
    # dataX_shape: shape of the space time image (1, 8377, 224, 224, 3)
    (num_frames, frame_h, frame_w, channels) = dataX_shape
    final_audio_vector = np.zeros((num_frames, audio_vectors.shape[0]))  # (1, 18, 8377)
    # Hanoi, you forgot to normalize the audio vectors in MATLAB!!!
    audio_vectors = audio_vectors/np.max(audio_vectors)  # normalize the audio vectors
    single_audio_vector = audio_vectors[:, 0:num_frames]  # Extract the corresponding audio vector, produces (18, 8377) from (18, 8379)
    final_audio_vector = single_audio_vector.T  # Assign the audio_vector to each video angle in idx=0 , (1, 8377, 224, 224, 3). Need to transpose it here.
    return final_audio_vector

