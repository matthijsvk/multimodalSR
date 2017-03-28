Phoneme Recognition using CTC
=====
Tensorflow implementation of Phoneme Recognition using Connectionist Temporal Classification. <http://www.cs.toronto.edu/~graves/icml_2006.pdf>. The model is created by bidirectional dynamic rnn with three stacked lstms in each direction.

Preprocessing
----
First, transform the Nist wav file in TIMIT dataset to normal wav file. I use the `scikits.audiolab` to finish this task.
`python phoneme_ctc.py transform -n train_dir -t test_dir -o out_dir`

Train
----
I extract the mfcc features using the module `python_speech_features`.
The dimension of feature is 39 and there are 39 classes for output phoneme labels. 
The work flow is:
extracted features from wav audio -> network -> phoneme labels.
After 100 epoches, which costs me 4 days using NVIDIA1070 GPU, the cost decrease from 120 to 40 and the label error rate from 90% to 30%.

`python phoneme_ctc.py train -n processed_train_dir -t processed_test_dir -o checkpoint_dir`

Decode
----
`python phoneme_ctc.py decode -m checkpoint_dir`. Then you can provide a wav audio file and the model will output the phoneme labels. Notce: the wav file should be 1 channel, 16 bits and bitrate 16000. I record the wav file using sox on my mac. e.g. `sox -d -b 16 -c 1 -r 16000 helloworld.wav`

I provide a trained model, you can test the model using `python phoneme_ctc.py decode -m model`.


Dependencies
----
`python 2.7`
`brew install libsndfile`
`pip install -r requirements.txt`

