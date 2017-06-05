function freq_vectors = computeAudioVectors(x, video_fps, audio_fps, num_vid_frames, num_cnn_frames)
    % Calculate audio block_size that corresponds to 1 video frame
    block_size = (audio_fps/video_fps);
    
    % Load the marimba fundamental frequencies (note_freq) and ranges
    % (note_freq_range)
    load marimba_freq_and_ranges.mat;
    
    % Normalize the audio signal to 1 and convert to mono
    x_norm = mean(x,2)/max(mean(x,2));
    
    %  We concatenate 3 frames of video, so we take 3x1920 frames in the audio
    % domain. The hops size is simply the block_size, since we "slide" by
    % 1920 each time.
    hop_size = block_size;
    
    % Need to pad tail of signal with some extra zeros so that when the
    % num_slides reaches the end, we can still compute fft
    zero_pad = zeros(block_size*3,1);
    x_norm = [x_norm; zero_pad];
    
    % Calculate number of valid slides
    % W_ = 1 + (W - WW)/stride
    num_windows = (1 + (num_vid_frames - num_cnn_frames));
    
    % Preallocate for speed. The number of 18_dim vectors is equal to number of valid slides.
    freq_vectors = zeros(18,num_windows);

    %m=30;
    for i = 1:num_windows
        %subplot(m,1,i)
        left_idx = 1 + (i*hop_size);
        right_idx = 1 + (i*hop_size) + (block_size*3)  % Need to put right index 3xblock_size away from left index
       
        % Extract the signal_segment from original signal
        signal_segment = x_norm(left_idx:right_idx);
        hamming_window = hamming(length(signal_segment));  % Generate hamming window of equal length
        signal_segment = times(signal_segment,hamming_window);  % Pointwise multiplication (applies the hamming window)
        %plot(signal_segment)
        
        % Artifically pad the signal_segment with zeros so that we get
        % 44100Hz resolution when doing FFT
        signal_segment_padded = padarray(signal_segment,[44100-length(signal_segment) 0], 0, 'post');

        Y = fft(signal_segment_padded);
        L = length(signal_segment_padded);

        P2 = abs(Y/L);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        
        %{
        f = Fs*(0:(L/2))/L;
        plot(f(1:3000),P1(1:3000))
        title('Single-Sided Amplitude Spectrum of X(t)')
        xlabel('f (Hz)')
        ylabel('|P1(f)|')
        hold on
        %}
        
        % For this particular window, define a single "slice" of freq
        % vector
        window_freq_vector = zeros(length(note_freq), 1);
        
        % Iterate through each of the 18 notes. For each note, look within
        % the range of frequencies defined in note_freq_range and extract
        % the frequency with the highest amplitude
        % 
        for j = 1:length(note_freq)
           freq_window_left = note_freq(j) - note_freq_range(j,1);
           freq_window_right = note_freq(j) + note_freq_range(j,2);
           freq_window = P1(round(freq_window_left): round(freq_window_right));
           %[pks,locs] = findpeaks(window);
           [M, I] = max(freq_window);
           % M = max(window);
           window_freq_vector(j) = M;
           %plot(window_left + f(I), M, 'or');
        end

        %freq_vectors = horzcat(freq_vectors, frame_freq_vector);
        freq_vectors(:,i) = window_freq_vector;
    end
    
end