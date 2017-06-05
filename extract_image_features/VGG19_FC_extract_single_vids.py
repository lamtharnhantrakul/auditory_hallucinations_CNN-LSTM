from extract_image_features.video_utils import *
import numpy as np
from extract_image_features.keras_pretrained_models.imagenet_utils import preprocess_input
from keras.models import Model
from keras.preprocessing import image
from extract_image_features.keras_pretrained_models.vgg19 import VGG19

# file saving and loading destinations change whether you are working on laptop or desktop
USE_TITANX = True
### CHANGE THE FILE TO BE READ HERE!!!!


######## LOADING VIDEO FILENAMES
print ("--- Loading video and audio filenames...")
if USE_TITANX:
    video_dir = '/home/zanoi/ZANOI/auditory_hallucination_videos'
else: # Working on MacBook Pro
    video_dir = "/Volumes/SAMSUNG_SSD_256GB/ADV_CV/2-25_VIDAUD/EXPORTS"

video_files = [os.path.join(video_dir, file_i)
         for file_i in os.listdir(video_dir)
         if file_i.endswith('.mp4')]

num_videos = len(video_files)
print("num_videos: ", num_videos)

######## LOADING AUDIO FILENAMES
audio_feature_dir = "../audio_vectors"

audio_f_files = [os.path.join(audio_feature_dir, file_i)
                for file_i in os.listdir(audio_feature_dir)
                if file_i.endswith('.mat')]

num_audio_f = len(audio_f_files)
print (audio_f_files)
print("num_audio_f: ", num_audio_f)


for audio_idx in range(num_audio_f):  # Loop over all audio files
    audio_prefix, audio_vector_length, audio_features = returnAudioVectors(audio_idx, audio_f_files)

    # Find all the linked videos for the given audio vector
    linked_video_f = findMatchingVideos(audio_prefix, video_files)
    print(audio_f_files[audio_idx])
    print(linked_video_f)

    for video_filename in linked_video_f:
        # Return the angle_name to name the file correctly
        angle_name = returnAngleName(video_filename)
        print ("angle_name:", angle_name)

        # Process the videos linked to a particular audio vector
        ######## PROCESS VIDEO TO BLACK AND WHITE
        print("--- Processing video to greyscale...")
        processed_video = processOneVideo(audio_vector_length, video_filename, normalize=False)
        print("processed_video.shape:", processed_video.shape)

        ######### CONCATENATE INTO SPACETIME IMAGE
        print ("--- Concatenating into Spacetime image...")
        window = 3
        space_time_image = createSpaceTimeImagesforOneVideo(processed_video,window) # (1, 8377, 224, 224, 3)
        print ("space_time_image.shape:", space_time_image.shape)

        ########## RUN THE SPACETIME IMAGES THROUGH VGG19
        print ("--- Running through VGG19 FC2 layer...")
        # Build the model
        base_model = VGG19(weights='imagenet')
        model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)  # Only take the FC2 layer output

        # Preallocate matrix output
        (num_frames, frame_h, frame_w, channels) = space_time_image.shape
        CNN_FC_output = np.zeros((num_frames,1,4096))  # (1,8377,1,4096) -> FC2 outputs dimensions (1,4096)

        for frame_num in tqdm(range(num_frames)):
            img = space_time_image[frame_num]
            x = np.expand_dims(img, axis=0)
            x = preprocess_input(x)
            fc2_features = model.predict(x)  # Predict the FC2 features from VGG19, output shape is (1,4096)

            CNN_FC_output[frame_num] = fc2_features  # Save the FC2 features to a matrix
        print("CNN_FC_output.shape:", CNN_FC_output.shape) # (1,8377,1,4096)

        ########### CREATE FINAL DATASET, concatenate FC output with audio vectors
        # Normalization of the audio_vectors occurs in this function -> Hanoi forgot to normalize in MATLAB!!!!
        final_audio_vector = createAudioVectorDatasetForOneVid(audio_features, space_time_image.shape) #(8377, 18)
        print ("final_audio_vector.shape:", final_audio_vector.shape)

        ############ PACKAGE AND SAVE THE DATASET
        if USE_TITANX:
            data_extern_dest = '/home/zanoi/ZANOI/auditory_hallucinations_data/FC_2_data/'
        else:  # Working on MacBook Pro
            data_extern_dest = '/Volumes/SAMSUNG_SSD_256GB/ADV_CV/data/'
        file_name = data_extern_dest + audio_prefix + angle_name + '_dataX_dataY.h5'

        with h5py.File(file_name, 'w') as hf:
            print ("Writing data to file...")
            hf.create_dataset('dataX', data=CNN_FC_output)
            hf.create_dataset('dataY', data=final_audio_vector)

print ("--- {EVERYTHING COMPLETE HOMIEEEEEEEEE} ---")