clear;
close all;
clc;

% Specify the folder where the files live. Right now they are all on my
% Samsung External SSD.
myFolder = '/Volumes/SAMSUNG_SSD_256GB/ADV_CV/2-25_VIDAUD/EXPORTS';
[video_file_names, audio_file_names] = retrieveFileNames(myFolder);

video_file_names

%Read video file
vidObj = VideoReader(char(video_file_names(7)));
video_fps = vidObj.FrameRate;

% Count the number of frames in the video file (must manually count,
% since the MATLAB video object is not 100% accurate)
num_vid_frames = 0;
while hasFrame(vidObj)
      readFrame(vidObj);
      num_vid_frames = num_vid_frames + 1;
end
num_vid_frames
numFrames = ceil(vidObj.FrameRate*vidObj.Duration)