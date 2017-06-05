from extract_image_features.video_utils import *
import matplotlib.pyplot as plt

### SET TO TRUE IF USING TITANX LINUX MACHINE
USE_TITANX = True

### Define video processing dimensions
frame_h = 170  # (60,60) maybe too small
frame_w = frame_h

### LOADING VIDEOS ###
print ("--- Loading video and audio filenames...")
if USE_TITANX:
    video_dir = '/home/zanoi/ZANOI/auditory_hallucinations_videos'
else: # Working on MacBook Pro
    video_dir = "/Volumes/SAMSUNG_SSD_256GB/ADV_CV/4-24_VIDAUD/EXPORTS"

video_files = [os.path.join(video_dir, file_i)
         for file_i in os.listdir(video_dir)
         if file_i.endswith('.mp4')]

num_videos = len(video_files)
print("num_videos: ", num_videos)

video_filename = video_files[0]
audio_f_length = 200
output_dimensions = (frame_h, frame_w)

greyscale_vid = processOneVideo(audio_f_length=audio_f_length, video_filename=video_filename,output_dimensions=output_dimensions,normalize=False)
print(greyscale_vid.shape)

num_images = 5


'''
for i in range(num_images):
    img = greyscale_vid[i]
    plt.subplot(num_images,1,i+1)
    imgplot = plt.imshow(img, cmap='gray')
plt.show()
'''

'''
def createSpaceTimeImagesforOneVideoCHECK(video, CNN_window):
    # video: single video of BW images
    # video.shape: (8379, 224, 224)
    (num_frames, frame_h, frame_w) = video.shape
    space_time_single_vid = np.zeros((num_frames - (CNN_window-1), frame_h,frame_w,CNN_window))  # (8377, 224, 224, 3)
    for i in tqdm(range(num_frames - (CNN_window-1))):
        l_idx = i
        r_idx = l_idx + (CNN_window)
        curr_stack = video[l_idx:r_idx, :, :] # Extracts (3,224,224)
        #print ("curr_stack.shape before reshape:", curr_stack.shape)


        for i in range(3):
            img = curr_stack[i, :, :]
            plt.subplot(3, 1, i + 1)
            plt.imshow(img, cmap='gray')
        plt.show()


        new_stack = np.zeros((frame_h,frame_w,CNN_window))
        for j in range(CNN_window):
            new_stack[:,:,j] = curr_stack[j,:,:]

        #curr_stack = curr_stack.reshape(frame_h,frame_w,CNN_window)  # reshapes to (224,224,3)
       # print("curr_stack.shape after reshape:", curr_stack.shape)

        for i in range(3):
            img = new_stack[:, :, i]
            plt.subplot(3, 1, i + 1)
            plt.imshow(img, cmap='gray')
        plt.show()



        space_time_single_vid[i,:,:,:] = new_stack
    return space_time_single_vid
'''

space_time_image = createSpaceTimeImagesforOneVideo(greyscale_vid, 3)
print(space_time_image.shape)

for i in range(3):
    img = space_time_image[18,:,:,i]
    plt.subplot(3,1,i+1)
    imgplot = plt.imshow(img, cmap='gray')
plt.show()

