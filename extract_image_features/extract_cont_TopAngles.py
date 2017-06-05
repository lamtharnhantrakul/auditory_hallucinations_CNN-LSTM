from extract_image_features.video_utils import *
import numpy as np
from keras_pretrained_models.imagenet_utils import preprocess_input
from keras.models import Model
from keras.preprocessing import image
from keras_pretrained_models.vgg19 import VGG19

########
### THIS CODE MANUALLY CONCATENATES THE 3 TOP ANGLE VIDEOS, it is not general purpose. Bad coding...Hanoi


######## LOADING VIDEO FILENAMES
USE_TITANX = True

# Define the external SSD where the dataset residesin
if USE_TITANX:
    data_dir = '/home/zanoi/ZANOI/auditory_hallucination_videos/'
else:
    data_dir = '/Volumes/SAMSUNG_SSD_256GB/ADV_CV/data/'

top_angles = ['seq1_angle5.mp4',
              'seq2_angle5.mp4',
              'seq3_angle4.mp4']

video_files = [data_dir + file_i for file_i in top_angles]

num_videos = len(video_files)
print("num_videos: ", num_videos)

######## LOADING AUDIO FILENAMES
audio_feature_dir = "../audio_vectors/"

top_vectors = ['seq1_audio_vectors.mat',
               'seq2_audio_vectors.mat',
               'seq3_audio_vectors.mat']

audio_f_files = [audio_feature_dir + file_i for file_i in top_vectors]

num_audio_f = len(audio_f_files)
print (audio_f_files)
print("num_audio_f: ", num_audio_f)

dataX = []
dataY = []

for idx in range(num_audio_f):  # Loop over all audio files
    print ("------- { Processing " + str(idx) + " } ---------")
    audio_prefix, audio_vector_length, audio_features = returnAudioVectors(idx, audio_f_files)

    # Process the videos linked to a particular audio vector
    ######## PROCESS VIDEO TO BLACK AND WHITE
    print("--- Processing video to greyscale...")
    processed_video = processOneVideo(audio_vector_length, video_filename=video_files[idx], normalize=False)
    print("processed_video.shape:", processed_video.shape)

    ######### CONCATENATE INTO SPACETIME IMAGE
    print ("--- Concatenating into Spacetime image...")
    window = 3
    space_time_image = createSpaceTimeImagesforOneVideo(processed_video,window) # (1, 8377, 224, 224, 3)
    print ("space_time_image.shape:", space_time_image.shape)

    dataX.append(space_time_image)  # This will be done 3 times, we will join them outside the loop

    ########### CREATE FINAL DATASET, concatenate FC output with audio vectors
    # Normalization of the audio_vectors occurs in this function -> Hanoi forgot to normalize in MATLAB!!!!
    final_audio_vector = createAudioVectorDatasetForOneVid(audio_features, space_time_image.shape) #(8377, 18)
    print ("final_audio_vector.shape:", final_audio_vector.shape)
    dataY.append(final_audio_vector)

############ JOIN THE VIDEOS INTO ONE LONG SEQUENCE
# manual code, REALLY BAD!
final_dataX = np.concatenate((dataX[0],dataX[1]), axis=0)
final_dataX = np.concatenate((final_dataX,dataX[2]), axis=0)

final_dataY = np.concatenate((dataY[0],dataY[1]), axis=0)
final_dataY = np.concatenate((final_dataY,dataY[2]), axis=0)

print ("final_dataX.shape:", final_dataX.shape)
print ("final_dataY.shape:", final_dataY.shape)

############ PACKAGE AND SAVE THE DATASET
if USE_TITANX:
    data_extern_dest = '/home/zanoi/ZANOI/auditory_hallucinations_data/TopAngle_data/'
else:  # Working on MacBook Pro
    data_extern_dest = '/Volumes/SAMSUNG_SSD_256GB/ADV_CV/data/'
file_name = data_extern_dest + 'TopAngle_dataX_dataY.h5'

with h5py.File(file_name, 'w') as hf:
    print ("Writing data to file...")
    hf.create_dataset('dataX', data=final_dataX)
    hf.create_dataset('dataY', data=final_dataY)

print ("--- {EVERYTHING COMPLETE HOMIEEEEEEEEE} ---")