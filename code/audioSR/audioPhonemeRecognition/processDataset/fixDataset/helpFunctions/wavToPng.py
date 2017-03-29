######################################
##############  FFT ##################
######################################
# generates PNG with time and frequency spectrum of the wav file
from numpy import linspace
from pylab import plot, savefig, show, subplot, xlabel, ylabel
from scipy import arange, fft
from scipy.io.wavfile import read


def plotSpectru(y, Fs):
    n = len(y)  # lungime semnal
    k = arange(n)
    T = n / Fs
    frq = k / T  # two sides frequency range
    frq = frq[range(n / 2)]  # one side frequency range

    Y = fft(y) / n  # fft computing and normalization
    Y = Y[range(n / 2)]
    return frq, abs(Y)


def wavToPng(filename):
    Fs, data = read(filename)
    y = data  # [:, 1]  #TODO for some WAV files read() produces 2 dimensional array, we only need one dim.
    timp = len(y) / float(Fs)
    t = linspace(0, timp, len(y))

    subplot(2, 1, 1)
    plot(t, y)
    xlabel('Time')
    ylabel('Amplitude')

    subplot(2, 1, 2)
    frq, mag = plotSpectru(y, Fs)
    plot(frq, mag, 'r')  # plotting the spectrum
    xlabel('Freq (Hz)')
    ylabel('|Y(freq)|')

    savefig(filename + 'brol.png', bbox_inches='tight')
    show()


wavToPng('sa1.wav')
