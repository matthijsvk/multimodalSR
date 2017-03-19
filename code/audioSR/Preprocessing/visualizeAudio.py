import matplotlib.pyplot as plt
from scipy.io import wavfile # get the api
from scipy.fftpack import fft
from pylab import *

def f(filename):
    fs, data = wavfile.read(filename) # load the data
    print(data.shape)
    a = data#.T[0] # this is a two channel soundtrack, I get the first track
    print(a.size)
    print("Normalizing...")
    b=[(ele/2**8.)*2-1 for ele in a] # this is 16-bit track, b is now normalized on [-1,1)
    print("Calculating FFT...")
    c = fft(b) # create a list of complex number
    d = len(c)/2  # you only need half of the fft list
    print("Plotting...")
    plt.plot(abs(c[:(d-1)]),'r')
    savefig(filename+'.png',bbox_inches='tight')


import glob
files = glob.glob('./*.wav')
for ele in files:
    f(ele)
quit()