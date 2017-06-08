# auditory-hallucinations
Check out the demo video!

# Overview
This projects explores cross-modal learning from the visual domain to the audio domain in the context of musical instruments. A neural network is trained on pairs of image sequences of a marimba being struck and the resulting sonic output.
Given a series of input images I<sub>1</sub>,I<sub>2</sub>,I<sub>3</sub>,...,I<sub>n</sub>, the neural network is tasked with producing the corresponding audio feature s<sub>1</sub>,s<sub>2</sub>,s<sub>3</sub>,...,s<sub>n</sub> where st ∈ R<sup>18</sup>. We initially approached this as a regression problem using an RNN that takes in CNN features as input.

The project was  inspired by the work "Visually Indicated Sounds" by Owens et al [1](http://vis.csail.mit.edu). However, unlike Owens et al, which worked with <i>unpitched</i> sounds from Nature and everyday objects composed of mainly <i>filtered white noise and impulses</i>, this work focuses on <b>pitched</b> musical instruments, specifically the marimba, that are <b>spectrally pure</b>.

At the time of writing, the CNN-LSTM is able to overfit a small sequence, but is not yet able to generalize to the entire dataset. 

# GT Marimba Dataset
With a plethora of musical instruments to pick from, we choose the marimba for a number of compelling reasons:
* Pitch information is completely encoded in the spatial domain (notes on the left are lower, notes on the right are higher)
* Pitch information is encoded in the geometry of the bars (bigger bars = lower pitch, smaller bars = higher pitch)
* Timbre information is encoded in the texture and color of the bars (wood vs metal texture)
* The marimba is a spectrally pure instrument, dominated by a couple of sinusoids and very few overtones.

We use a stereo XY condensor microphone and an array of video cameras to capture a marimba being played. This provides approximately 6 hours of footage or roughly 850,000 frames. In every shot, we choose angles that maintain the full marimba keys to preserve relative spatial information.

We also provide an extended dataset of only top angle footage with small offsets. This was easiest angle to train the neural network on. This amounts to approximately 3 hours of footage or 500,000 frames. 

# Image Features
A "space-time" image is produced 

# Audio Features
For a complete description, see the paper linked below. Briefly, for every CNN window


# Architectures

# Results

# Credits
* Deep Learning: Lamtharn (Hanoi) Hantrakul 
* Audio DSP: Lamtharn (Hanoi) Hantrakul 
* Image Processing: Lamtharn (Hanoi) Hantrakul & Si Chen
* Advisor: Prof. James Hays
* Special thanks to Dr. Mason Bretan for insightful discussion and advice throughout the project. 
