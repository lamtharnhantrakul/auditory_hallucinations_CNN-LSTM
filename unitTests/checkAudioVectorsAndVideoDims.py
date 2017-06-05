from extract_image_features.video_utils import *
import numpy as np

# This code checks the dimensions of the audio_vectors and the dimensions of the video to see if they agree

USE_TITANX = False

### LOADING VIDEOS ###
print ("--- Loading video and audio filenames...")
if USE_TITANX:
    video_dir = '/home/zanoi/ZANOI/auditory_hallucination_videos'
else: # Working on MacBook Pro
    video_dir = "/Volumes/SAMSUNG_SSD_256GB/ADV_CV/4-24_VIDAUD/EXPORTS"

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

for i in range(num_audio_f):  # Loop over all audio files
    print ("-----------------------------------------------")
    audio_prefix, audio_vector_length, audio_features = returnAudioVectors(i, audio_f_files)

    # Find all the linked videos for the given audio vector
    linked_video_f = findMatchingVideos(audio_prefix, video_files)
    print(audio_f_files[i])
    print(linked_video_f)

    for video_filename in linked_video_f:
        vid = imageio.get_reader(video_filename, 'ffmpeg')
        num_frames = 0
        for i in tqdm(range(audio_vector_length + 1)):
            with warnings.catch_warnings():  # Ignores the warnings about depacrated functions that don't apply to this code
                warnings.simplefilter("ignore")
                img = vid.get_data(i)
            num_frames += 1
        print ("num_frames:",num_frames)

print ("----{ CHECKING COMPLETE }-----")