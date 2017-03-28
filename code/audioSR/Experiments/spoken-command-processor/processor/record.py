import struct
import itertools
import wave
import numpy as np
import scipy
import librosa
import pyaudio
from sklearn import preprocessing
from model import speech2phonemes, dataset, utils

# Constants
WIDTH = 2           # bytes per sample
CHANNELS = 1        # mono
RATE = 16000      	# Sampling rate (samples/second)
BLOCKSIZE = 1024
DURATION = 1        # Duration in seconds
BLOCKS = int(DURATION * RATE / BLOCKSIZE)
THRESHOLD = 1000

def record_input(save=True, wavfile="input.wav"):
    # Open audio device
    p = pyaudio.PyAudio()

    fmt = p.get_format_from_width(WIDTH)
    stream = p.open(
        format=fmt,
        channels=CHANNELS,
        rate=RATE,
        frames_per_buffer=BLOCKSIZE,
        input=True,
        output=False
    )

    # block reading
    bn, start_rec = 0, False
    frames = []
    print(">>> start recording...")
    while bn < BLOCKS:
        # Read audio by block, convert
        input_string = stream.read(BLOCKSIZE)
        input_tuple = struct.unpack('h'*BLOCKSIZE, input_string)
        # if input not loud enough, ignore
        if not start_rec and max(input_tuple) > THRESHOLD:
            start_rec = True
            print(">>> threshold met!")

        if start_rec:
            frames.append(input_string)
            bn += 1

    print(">>> finish record.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    if save:
        write_wavfile(frames, wavfile)
    return frames, wavfile

def write_wavfile(frames, wavfile):
    wf = wave.open(wavfile, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(WIDTH)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

if __name__ == "__main__":
    record_input()
