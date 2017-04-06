'''
Created on Apr 7, 2011
by Uri Nieto
uri@urinieto.com

Sample Rate converter from 48kHz to 44.1kHz.

USAGE:
$>python resample.py -i input.wav [-o output.wav -q [0.0-1.0]]

EXAMPLES:
$>python resample.py -i onades.wav
$>python resample.py -i onades.wav -o out.wav
$>python resample.py -i onades-mono.wav -q 0.8
$>python resample.py -i onades.wav -o out3.wav -q 0.5

DESCRIPTION:
The input has to be a WAV file sampled at 48 kHz/sec
with a resolution of 16 bits/sample. It can have n>0
number of channels (i.e. 1=mono, 2=stereo, ...).

The output will be a WAV file sampled at 44.1 kHz/sec
with a resolution of 16 bits/sample, and the same
number of channels as the input.

A quality parameter q can be provided (>0.0 to 1.0), and
it will modify the number of zero crossings of the filter, 
making the output quality best when q = 1.0 and very bad 
as q tends to 0.0

The sample rate factor is:

 44100     147
------- = ----- 
 48000     160
 
To do the conversion, we upsample by 147, low pass filter, 
and downsample by 160 (in this order). This is done by
using an efficient polyphase filter bank with resampling 
algorithm proposed by Vaidyanathan in [2].

The low pass filter is an impulse response windowed
by a Kaiser window to have a better filtering 
(around -60dB in the rejection band) [1].

As a comparison between the Kaiser Window and the
Rectangular Window, this algorithm plotted the following
images included with this package:

KaiserIR.png
KaiserFR.png
RectIR.png
RectFR.png

The images show the Impulse Responses and the Frequency
Responses of the Kaiser and Rectangular Windows. As it can be
clearly seen, the Kaiser window has a gain of around -60dB in the
rejection band, whereas the Rect window has a gain of around -20dB
and smoothly decreasing to -40dB. Thus, the Kaiser window
method is rejecting the aliasing much better than the Rect window.

The Filter Design is performed in the function:
designFIR()

The Upsampling, Filtering, and Downsampling is
performed in the function:
upSampleFilterDownSample()

Also included in the package are two wav files sampled at 48kHz
with 16bits/sample resolution. One is stereo and the other mono:

onades.wav
onades-mono.wav

NOTES:
You need numpy and scipy installed to run this script.
You can find them here:
http://numpy.scipy.org/

You may want to have matplotlib as well if you want to 
print the plots by yourself (commented right now)

This code would be much faster on C or C++, but my decision
on using Python was to make the code more readable (yet useful)
rather than focusing on performance.

@author: uri

REFERENCES:
[1]: Smith, J.O., "Spectral Audio Signal Processing", 
W3K Publishing, March 2010

[2] Vaidyanathan, P.P., "Multirate Systems and Filter Banks", 
Prentice Hall, 1993.

COPYRIGHT NOTES:

Copyright (C) 2011, Uri Nieto

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
 
'''

import sys
import time as tm
from decimal import Decimal
from fractions import Fraction
from shutil import copyfile

import numpy as np
from scipy.io import wavfile

# import matplotlib.pyplot as plt #Uncomment to plot


'''
h = flipTranspose(h, L, cpp)
...
Desc: Flips and Transposes the impulse response h, dividing
it into L different phases with cpp coefficients per phase.
Following as described in [2].
...
h: Impulse response to flip and Transpose
L: Upsampling Factor
cpp: Coeficients per Phase
return hh: h flipped and transposed following the descritpion
'''


def flipTranspose(h, L, cpp):
    # Get the impulse response size
    N = len(h)

    # Init the output to 0
    hh = np.zeros(N)

    # Flip and Transpose:
    for i in range(L):
        hh[cpp - 1 + i * cpp:-N - 1 + i * cpp:-1] = h[i:cpp * L:L]

    return hh


'''
h = upSampleFilterDownSample(x, h, L, M)
...
Desc: Upsamples the input x by L, filters it out using h, and
downsamples it by M.

The algorithm is based on the "efficient polyphase filter bank 
with resampling" found on page 129 of the book [2] (Figure 4.3-8d).

...
x: input signal
h: impulse response (assumes it has the correct cut-off freq)
L: Upsampling Factor
M: Downsampling Factor
returns y: output signal (x upsampled, filtered, and downsampled)
'''


def upSampleFilterDownSample(x, h, L, M, printing=False):
    # Number of samples to convert
    N = len(x)

    # Compute the number of coefficients per phase
    cpp = len(h) / L

    # Flip and Transpose the impulse response
    h = flipTranspose(h, L, cpp)

    # Check number of channels
    if (np.shape(np.shape(x)) == (2,)):
        nchan = np.shape(x)[1]
        y = np.zeros(int((np.ceil(N * L / float(M)), nchan)))
    else:
        nchan = 1
        y = np.zeros(int(np.ceil(N * L / float(M))))

    # Init the output index
    y_i = 0

    # Init the phase index
    phase_i = 0

    # Init the main loop index
    i = 0

    # Main Loop
    while i < N:

        # Print % completed
        if (printing and (i % 30000 == 0)):
            print("%.2f %% completed" % float(100 * i / float(len(x))))

        # Compute the filter index
        h_i = phase_i * cpp

        # Compute the input index
        x_i = i - cpp + 1;

        # Update impulse index if needed (offset)
        if x_i < 0:
            h_i -= x_i
            x_i = 0

        # Compute the current output sample
        rang = i - x_i + 1
        if nchan == 1:
            y[y_i] = np.sum(x[x_i:x_i + rang] * h[h_i:h_i + rang])
        else:
            for c in range(nchan):
                y[y_i, c] = np.sum(x[x_i:x_i + rang, c] * h[h_i:h_i + rang])

                # Add the downsampling factor to the phase index
        phase_i += M

        # Compute the increment for the index of x with the new phase
        x_incr = phase_i / int(L)

        # Update phase index
        phase_i %= L

        # Update the main loop index
        i += x_incr

        # Update the output index
        y_i += 1

    return y


'''
h = impulse(M, L)
...
M: Impulse Response Size
T: Sampling Period
returns h: The impulse response
'''


def impulse(M, T):
    # Create time array
    n = np.arange(-(M - 1) / 2, (M - 1) / 2 + 1)

    # Compute the impulse response using the sinc function
    h = (1 / T) * np.sinc((1 / T) * n)

    return h


'''
b = bessel(x)
...
Desc: Zero-order modified Bessel function of the first kind, with
approximation using the Maclaurin series, as described in [1]
...
x: Input sample
b: Zero-order modified Bessel function of the first kind
'''


def bessel(x):
    return np.power(np.exp(x), 2);


'''
k = kaiser(M, beta)
...
Desc: Generates an M length Kaiser window with the
specified beta parameter. Following instructions in [1]
...
M: Number of samples of the window
beta: Beta parameter of the Kaiser Window
k: array(M,1) containing the Kaiser window with the specified beta
'''


def kaiser(M, beta):
    # Init Kaiser Window
    k = np.zeros(M)

    # Compute each sample of the Kaiser Window
    i = 0
    for n in np.arange(-(M - 1) / 2, (M - 1) / 2 + 1):
        samp = beta * np.sqrt(1 - np.power((n / (M / 2.0)), 2))
        samp = bessel(samp) / float(bessel(beta))
        k[i] = samp
        i = i + 1

    return k


'''
h = designFIR(N, L, M)
...
Desc: Designs a low pass filter to perform the conversion of 
sampling frequencies given the upsampling and downsampling factors.
It uses the Kaiser window to better filter out aliasing.
...
N: Maximum size of the Impulse Response of the FIR
L: Upsampling Factor
M: Downsampling Factor
returns h: Impulse Response of the FIR
'''


def designFIR(N, L, M):
    # Get the impulse response with the right Sampling Period
    h0 = impulse(N, float(M))

    # Compute a Kaiser Window
    alpha = 2.5  # Alpha factor for the Kaiser Window
    k = kaiser(N, alpha * np.pi)

    # Window the impulse response with the Kaiser window
    h = h0 * k

    # Filter Gain
    h = h * L

    # Reduce window by removing almost 0 values to improve filtering
    for i in range(len(h)):
        if abs(h[i]) > 1e-3:
            for j in range(i, 0, -1):
                if abs(h[j]) < 1e-7:
                    h = h[j:len(h) - j]
                    break
            break

    '''
    # For plotting purposes:
    N = len(h)
    Hww = fft(h, N)
    Hww = Hww[0:N/2.0]
    Hwwdb = 20.0*np.log10(np.abs(Hww))
    Hw = fft(h0, N)
    Hw = Hw[0:N/2.0]
    Hwdb = 20.0*np.log10(np.abs(Hw))

    plt.figure(1)
    plt.plot(h)
    plt.title('Kaiser Windowed Impulse Response of the low pass filter')
    plt.show()
    
    plt.figure(2)
    plt.plot(h0)
    plt.title('Rect Windowd Impulse Response of the low pass filter')
    plt.show()
    
    #print np.shape(np.arange(0, N/2.0-1)/float(N)), np.shape(Hwwdb)
    
    plt.figure(3)
    plt.plot(np.arange(0, N/2.0-1)/float(N),Hwwdb)
    plt.xlabel('Normalized Frequency');
    plt.ylabel('Magnitude (dB)');
    plt.title('Amplitude Response, Kaiser window with beta = ' +  str(2.5*np.pi));
    plt.show()
    
    plt.figure(4)
    plt.plot(np.arange(0, N/2.0-1)/float(N),Hwdb)
    plt.xlabel('Normalized Frequency');
    plt.ylabel('Magnitude (dB)');
    plt.title('Amplitude Response using a Rect Window');
    plt.show()
    '''

    return h


'''
dieWithUsage()
...
Desc: Stops program and prints usage
'''


def dieWithUsage():
    usage = """
    USAGE: $>python resample.py -i input.wav [-o output.wav -q (0.0-1.0]]

    input.wav has to be sampled at 48kHz with 16bits/sample
    If no output file is provided, file will be written in "./output.wav"
    If no quality param is provided,  1.0 (max) will be used
    
    Description: Converts sampling frequency of input.wav from 48kHz to 16kHz and writes it into output.wav
    """
    print
    usage
    sys.exit(1)


'''
Main
...
Desc: Reads the input wave file, designs the filter,
upsamples, filters, and downsamples the input, and
finally writes it to the output wave file.
'''


def resampleWAV(inFile, outFile="output.wav", out_fr=16000.0, q=0.0):
    # Parse arguments
    inPath, outPath, out_fr, q = inFile, outFile, out_fr, q

    # Read input wave
    in_fr, in_data = wavfile.read(inPath)
    in_nbits = in_data.dtype

    # Time it
    start_time = tm.time()

    # Set output wave parameters
    out_nbits = in_nbits

    frac = Fraction(Decimal(str(float(in_fr) / out_fr))).limit_denominator(1000)

    if (float(frac) < 1.0):
        print("input file smaller sampling rate than output..")
        print("input: ", in_fr, "Hz. Output: ", out_fr, "Hz")

    elif (float(frac) == 1.0):
        try:
            copyfile(inPath, outPath)
            print("Input file", inPath, "already at correct sampling rate; just copying")
        except Exception, e:
            print(e.args)
        else:
            print("some weird error", inPath, outPath)
            return -1
    else:
        L = frac.denominator  # Upsampling Factor
        M = frac.numerator  # Downsampling Factor
        Nz = int(25 * q)  # Max Number of Zero Crossings (depending on quality)
        h = designFIR(Nz * L, L, M)

        # Upsample, Filter, and Downsample
        out_data = upSampleFilterDownSample(in_data, h, L, M, printing=False)  # control progression output

        # Make sure the output is 16 bits
        out_data = out_data.astype(out_nbits)

        # Write the output wave
        wavfile.write(outPath, out_fr, out_data)

        # Print Results
        duration = float(tm.time() - start_time)
        print(
        "File", outPath, " was successfully written. Resampled from ", in_fr, "to", out_fr, "in", duration, " seconds")
    return 0
