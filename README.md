# Deep Auditory Hallucinations
Check out the demo video!

[![Video link](https://github.com/lamtharnhantrakul/auditory_hallucinations_CNN-LSTM/blob/master/assets/youtube.png)](https://www.youtube.com/watch?v=23lJOX4Ioo4&feature=youtu.be)

# Overview
This projects explores cross-modal learning from the visual domain to the audio domain in the context of musical instruments. A neural network is trained on pairs of image sequences of a marimba being struck and the resulting sonic output.
Given a series of input images I<sub>1</sub>,I<sub>2</sub>,I<sub>3</sub>,...,I<sub>n</sub>, the neural network is tasked with producing the corresponding audio feature s<sub>1</sub>,s<sub>2</sub>,s<sub>3</sub>,...,s<sub>n</sub> where st ∈ R<sup>18</sup>. We initially approached this as a regression problem using an RNN that takes in CNN features as input.

The project was inspired by the work "Visually Indicated Sounds" by [Owens et al](http://vis.csail.mit.edu). However, unlike Owens et al, which worked with <i>unpitched</i> sounds from Nature and everyday objects composed of mainly <i>filtered white noise and impulses</i>, this work focuses on <b>pitched</b> musical instruments, specifically the marimba, that are <b>spectrally pure</b>.

<img src="assets/pipeline.jpg" width="200" height="200" />

<b>At the time of writing, the CNN-LSTM is able to overfit a small sequence, but is not yet able to generalize to the entire dataset.</b>

# GT Marimba Dataset
With a plethora of musical instruments to pick from, we choose the marimba for a number of compelling reasons:
* Pitch information is completely encoded in the spatial domain (notes on the left are lower, notes on the right are higher)
* Pitch information is encoded in the geometry of the bars (bigger bars = lower pitch, smaller bars = higher pitch)
* Timbre information is encoded in the texture and color of the bars (wood vs metal texture)
* The marimba is a spectrally pure instrument, dominated by a couple of sinusoids and very few overtones.

<img src="assets/dataset.png" width="500" height="324" />

We use a stereo XY condensor microphone and an array of video cameras to capture a marimba being played. This provides approximately 6 hours of footage or roughly 850,000 frames. In every shot, we choose angles that maintain the full marimba keys to preserve relative spatial information.

We also provide an extended dataset of only top angle footage with small offsets. This was easiest angle to train the neural network on. This amounts to approximately 3 hours of footage or 500,000 frames. 

# Image Feature
<img src="assets/space_time.png" width="500" height="324" />

Like Owens et al, we use a "space-time" image consisting of 3 consecutive frames that have been grayscaled. An image vector I<sub>n</sub> is like a single image where each RGB channel is a greyscale image. 

# Audio Feature
For a more complete description, see the paper linked below. Briefly, to account for differences in the video and audio sampling rates (25fps vs 48000Hz or fps), we calculate the equivalent "window" in the audio domain for every 3 frames of concatenated space time image in the visual domain. We apply a Hamming window, artificially zero-pad the segment and take a Short Time Fourier Transform (STFT).

<img src="assets/sync_audio_video.png" width="40" height="324" />

Since the Marimba produces discrete, pitched notes, we can can map every frame of sound to a known note (essentially discretizing the frequency domain into 18 notes). At any time <i>t</i>, a single audio feature feature vector <b><i>s<sub>t</sub></i></b> is a R<sup>18</sup> vector. Each dimension of the vector contains the instantaneous amplitude of the frequency bin. A series of audio vectors thus captures how the amplitude of discrete frequency bins change over time. 

<img src="assets/audio_feature.png" width="500" height="300" />

Inspired by Owens et al paper, where the authors use both kNN and inverse synthesis over natural sounds, our audio feature also enables <i>both</i> nearest neighbour search and inverse synthesis of pitched marimba sounds. Briefly, the audio vector amplitudes can be interpolated and pointwise multiplied with 18 oscillators and summed via additive synthesis to reproduce the sound of the marimba!

# Architectures
<img src="assets/model_arch.png" width="500" height="400" />

We adopt 

# Results

# Dependancies

# Usage

# Credits
* Deep Learning: Lamtharn (Hanoi) Hantrakul 
* Audio DSP: Lamtharn (Hanoi) Hantrakul 
* Image Processing: Lamtharn (Hanoi) Hantrakul & Si Chen
* Advisor: Prof. James Hays
* Special thanks to Dr. Mason Bretan for insightful discussion and advice throughout the project. 
