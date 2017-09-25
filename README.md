
This is the repository containing most of the code for my thesis 'Design, Implementation and Analysis of a Deep Convolutional-Recurrent Neural Network for Speech Recognition throuth Audiovisual Sensor Fusion' at the ESAT (Electrical Engineering) Department of KU Leuven (2016-2017).  

Author: Matthijs Van keirsbilck  
Supervisor: Bert Moons  
Promotor: Marian Verhelst  

The code and thesis text are bound by the [KU Leuven's Student Thesis Copyright Regulations](https://admin.kuleuven.be/sab/jd/en/student-thesis-copyright).  
___


The CNN-LSTM networks for lipreading are combined with LSTM networks for audio recognition through an attention mechanism.   
These networks achieve state-of-the-art phoneme recognition performance on the publicly available audio-visual dataset [TCD-TIMIT](https://sigmedia.tcd.ie/TCDTIMIT/).
Systems that rely only audio suffer greatly when audio quality is lowered by noise, as is often the case in real-life situations.  
This performance loss can be greatly mitigated by adding visual information.  
The CNN-LSTM neural networks acieve 68.46% correctness compared to the 57.85% baseline.  
Audio-only neural networks achieve 67.03% compared to 65.47% in the baseline.  
Lipreading-audio combination networks achieve 75.70% accuracy for clean audio, and 58.55% for audio with an SNR of 0dB. The baseline multimodal network achieved 59% and 44% for clean and noisy audio, respectively.

___

The networks are implemented using [Lasagne](https://github.com/Lasagne).  
For the downloading, preprocessing etc of the dataset: see https://github.com/matthijsvk/TCDTIMITprocessing  
For the lipreading networks, see the folder `code/lipreading`   
For the audio speech recognition networks, see `code/audioSR`   
For the combination networks see `code/combinedSR`  


Thanks to the authors of all the data and software used in this work. An inexhaustive list:  
- Dataset: [TCDTIMIT](https://sigmedia.tcd.ie/TCDTIMIT/)  
- Frameworks:  
    [Theano](http://www.deeplearning.net/software/theano/)  
    [Lasagne](https://github.com/Lasagne)  
- Other Software:  
    - [Dlib](http://dlib.net/)  
    - [Attention Network in Lasagne](https://github.com/craffel/ff-attention)    
    - [Phoneme Classification on TIMIT](https://github.com/Faur/TIMIT)  
    - [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)  

To Set up Python, I recommend using [Anaconda](https://www.continuum.io/downloads). You can use the provided `environment.yml` to install all python packages (although some aren't used anymore).    
For the installation of Theano/Lasagne and CUDA, I recommend following [this tutorial](https://github.com/Lasagne/Lasagne/wiki/From-Zero-to-Lasagne-on-Ubuntu-14.04).



