%%% 12/3/2017
% This code generates the audio_vectors for all given audio files and their
% corresponding video files. All the video and audio files should be
% placed in one folder. At the moment this code points to a folder in an
% external SSD

clear;
close all;
clc;

% Specify the folder where the files live. Right now they are all on my
% Samsung External SSD.
%myFolder = '/Volumes/SAMSUNG_SSD_256GB/ADV_CV/2-25_VIDAUD/EXPORTS';

% Process second set of files filmed on 4/21/2017
myFolder = '/Volumes/SAMSUNG_SSD_256GB/ADV_CV/4-24_VIDAUD/EXPORTS';

[video_file_names, audio_file_names] = retrieveFileNames(myFolder);
% For every unique audio file name, find the corresponding video to
% calculate video2audio fps. Then use this information to calculate the
% audio vectors. 
for i = 1:length(audio_file_names)
    [video_file_name, seq_n] = retrieveVideoFileNameForAudio(video_file_names, audio_file_names(i));
    % Print out for double_checking
    audio_file_name = audio_file_names(i)
    video_file_name
    
    %Read video file
    vidObj = VideoReader(char(video_file_name));
    video_fps = vidObj.FrameRate;
    
    % Count the number of frames in the video file (must manually count,
    % since the MATLAB video object is not 100% accurate)
    num_vid_frames = 0;
    while hasFrame(vidObj)
          readFrame(vidObj);
          num_vid_frames = num_vid_frames + 1;
    end

    % Read audio file
    [x, audio_fps] = audioread(char(audio_file_name));
    num_cnn_frames = 3;

    % Compute the audio_vectors based on information on video and audio
    audio_vectors = computeAudioVectors(x, video_fps, audio_fps, num_vid_frames, num_cnn_frames);

    % Plot computed audio_vectors and original waveform
    plotAudioVector(x, audio_vectors, seq_n);
    
    % Save the audio vectors in the same directory
    file_name = strcat(seq_n,'_audio_vectors.mat');
    save(file_name, 'audio_vectors'); 
    
    % clear variable to save space 
    clear audio_vectors;
end




