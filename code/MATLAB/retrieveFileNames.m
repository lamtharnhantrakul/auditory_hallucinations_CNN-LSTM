function [video_file_names, audio_file_names] = retrieveFileNames(myFolder)

% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isdir(myFolder)
  errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
  uiwait(warndlg(errorMessage));
  return;
end

% Get a list of all the audio files
aud_file_pattern = fullfile(myFolder, '*.wav'); % Change to whatever pattern you need.
audio_files = dir(aud_file_pattern);
audio_file_names = cell(length(audio_files),1);
for k = 1 : length(audio_files)
  baseFileName = audio_files(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  %fprintf(1, 'Now reading %s\n', fullFileName);
  audio_file_names{k} = fullFileName;
end

% Get a list of all the video files
video_file_pattern = fullfile(myFolder, '*.mp4');
video_files = dir(video_file_pattern);
video_file_names = cell(length(video_files),1);
for k = 1 : length(video_files)
  baseFileName = video_files(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  %fprintf(1, 'Now reading %s\n', fullFileName);
  video_file_names{k} = fullFileName;
end

end

