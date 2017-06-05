function[video_file_name, seq_n] = retrieveVideoFileNameForAudio(video_file_names, audio_file_name)
    audio_file_name = char(audio_file_name);
    strings = strsplit(audio_file_name,'/');
    seq_name_temp = char(strings(end));
    seq_prefix_temp = strsplit(seq_name_temp,'_');
    seq_n = char(seq_prefix_temp(1));
  
    
    is_a_match = ~cellfun(@isempty,regexp(video_file_names, seq_n, 'match'));
    vector_of_indices = find(is_a_match);
    audio_matched_video_name = video_file_names(vector_of_indices);
    video_file_name = audio_matched_video_name(1);
end