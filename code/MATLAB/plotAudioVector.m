function plotAudioVector(x, audio_vectors, seq_n)
    figure
    subplot(2,1,1)
    x_norm = mean(x,2)/max(mean(x,2));
    zero_pad = zeros(1920*3,1);
    x_norm = [x_norm; zero_pad];
    plot(x_norm)

    max_audio_vector = max(audio_vectors(:));
    audio_vectors_norm = audio_vectors/max_audio_vector;
    title_name = strcat('Audio vector s(t) of signal:', seq_n);
    title(title_name)
    xlabel 'Time (s)'
    ylabel 'Amplitude'

    subplot(2,1,2)
    %image(audio_vectors/max_audio_vector,'CDataMapping','scaled')
    image(audio_vectors_norm, 'CDataMapping', 'scaled')
    colorbar
    title('Audio vector s(t) of signal')
    xlabel 'Time (s)'
    ylabel 'Note'
    ax = gca; 
    ax.YTick = ([1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18]);
    ax.YTickLabel = ({'F#4', 'G#4', 'Bb4', 'Db4', 'Eb4', 'F#5', 'G#5', 'Bb5', 'Db5', 'Eb5', 'F#6', 'G#6', 'Bb6', 'Db6', 'Eb6', 'F#7', 'G#7', 'Bb7'});
end