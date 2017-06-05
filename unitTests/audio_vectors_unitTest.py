import scipy.io as sio
import numpy as np
import os

### LOADING AUDIO ###
audio_feature_dir = "../audio_vectors"

audio_f_files = [os.path.join(audio_feature_dir, file_i)
                for file_i in os.listdir(audio_feature_dir)
                if file_i.endswith('.mat')]

num_audio_f = len(audio_f_files)
print("num_audio_f: ", num_audio_f)
for i in range(num_audio_f):
    print("------------------------------")
    audio_f_file = audio_f_files[i]  # Test with just one audio feature vector, and find all the corresponding movies
    print(audio_f_file)
    mat_contents = sio.loadmat(audio_f_file)  # 18 x n-2
    audio_vectors = mat_contents['audio_vectors']
    audio_vector_length = audio_vectors.shape[1]
    print("np.max(audio_vectors)", np.max(audio_vectors))
    print("np.min(audio_vectors)", np.min(audio_vectors))