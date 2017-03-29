import scipy.io.wavfile

wavfile = "/home/matthijs/TCDTIMIT/TIMIT/fixed/TIMIT/TRAIN/DR1/FCJF0/SA1.WAV"
sampling_rate, frames = scipy.io.wavfile.read(wavfile)
