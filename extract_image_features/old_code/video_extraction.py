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

### LOADING VIDEOS ###
video_dir = "/Volumes/SAMSUNG_SSD_256GB/ADV_CV/2-25_VIDAUD/EXPORTS"

video_files = [os.path.join(video_dir, file_i)
         for file_i in os.listdir(video_dir)
         if file_i.endswith('.mp4')]

num_videos = len(video_files)
print("num_videos: ", num_videos)


### LOADING AUDIO ###
audio_feature_dir = "../audio_vectors"

audio_f_files = [os.path.join(audio_feature_dir, file_i)
                for file_i in os.listdir(audio_feature_dir)
                if file_i.endswith('.mat')]

num_audio_f = len(audio_f_files)
print("num_audio_f: ", num_audio_f)


### HELPER FUNCTIONS ### -> need to move to a utils.py
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

def processOneVideo(audio_f_length, video_filename):
    # audio_f_length = int (length of audio feature vector corresponding to the video)
    # video_filename = str (filename of a single video)

    vid = imageio.get_reader(video_filename, 'ffmpeg')
    greyscale_vid = []
    for i in tqdm(range(audio_f_length + 1)):
        # apparently if I have an audio_vector of dimensions (18,8378), then the number of frames in the video is 8379
        with warnings.catch_warnings():  # Ignores the warnings about depacrated functions that don't apply to this code
            warnings.simplefilter("ignore")
            img = vid.get_data(i)
            img = img / np.max(img)  # rescale to 0 - 1 scale
            img = transform.resize(img, (224, 224), preserve_range=True)  # resize images to 224 x 224
            img = color.rgb2gray(img)  # convert to greyscale
            greyscale_vid.append(img)
    greyscale_vid = np.array(greyscale_vid)
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

### CHANGE THE FILE TO BE READ HERE!!!!
audio_idx = 1
audio_f_file = audio_f_files[audio_idx]  # Test with just one audio feature vector, and find all the corresponding movies
mat_contents = sio.loadmat(audio_f_file)  # 18 x n-2
audio_vectors = mat_contents['audio_vectors']
audio_vector_length = audio_vectors.shape[1]
#print(audio_f_files[0])
#print("audio_vectors.shape: ", audio_vectors.shape)

# Extract the file prefix using regular expressions
start = audio_f_file.find('seq')
end = audio_f_file.find("_audio", start)
audio_prefix = audio_f_file[start:end]

# Find all the linked videos for the given audio vector
linked_video_f = findMatchingVideos(audio_prefix, video_files)
print(audio_f_file)
print(linked_video_f)

# Process the videos linked to a particular audio vector
processed_videos = processVideos(audio_vector_length, linked_video_f)
print ("processed_videos.shape:", processed_videos.shape)

# Save the videos to an external file because the uncompressed file is HUGE
vid_extern_dest = '/Volumes/SAMSUNG_SSD_256GB/ADV_CV/videos_BW/'
file_name = vid_extern_dest + audio_prefix + '_vid_BW.h5'
with h5py.File(file_name, 'w') as hf:
    print ("Writing data to file")
    hf.create_dataset('videos_BW',  data=processed_videos)


with h5py.File(file_name, 'r') as hf:
    print("Reading data from file")
    video_data = hf['videos_BW'][:]
print("video_data.shape:", video_data.shape)







'''
print ("writing numpy array...")
np.save(file_name, processed_videos)
numpy_array = np.load(file_name)
print ("numpy_array.shape", numpy_array)
'''


'''
for audio_f_file in audio_f_files:  # enumerate the file string
    # Read in the audio vector file
    mat_contents = sio.loadmat(audio_f_file)  # 18 x n-2
    audio_vectors = mat_contents['audio_vectors']
    #print(audio_f_files[0])
    #print("audio_vectors.shape: ", audio_vectors.shape)

    # Extract the file prefix using regular expressions
    start = audio_f_file.find('seq')
    end = audio_f_file.find("_audio", start)
    audio_prefix = audio_f_file[start:end]

    # Find all the linked videos for the given audio vector
    linked_video_f = findMatchingVideos(audio_prefix, video_files)
    print(audio_f_file)
    print(linked_video_f)
'''



'''
vid = imageio.get_reader(video_files[0],  'ffmpeg')
num_frames = 0
for i in range(audio_vectors.shape[1] + 1):
    image = vid.get_data(i)
    print(i, image.shape)
'''



'''
for vid_idx in range(number_videos):
    # Read in audio data to labels
    mat_contents = sio.loadmat("../seq4_audio_vectors.mat")  # 18 x n-2
    audio_vectors = mat_contents['audio_vectors']
    print audio_vectors.shape
    Y_vid = np.matrix.transpose(audio_vectors)
    print Y_vid.shape  # should be previous shape but flipped indices
'''