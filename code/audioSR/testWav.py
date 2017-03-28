import numpy as np
import scikits.audiolab

data = np.random.uniform(-1,1,44100)
# write array to file:
scikits.audiolab.wavwrite(data, 'test.wav', fs=44100, enc='pcm16')
# play the array:
scikits.audiolab.play(data, fs=44100)
