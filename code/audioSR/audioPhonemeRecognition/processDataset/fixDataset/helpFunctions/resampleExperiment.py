######################################
########## RESAMPLING  ###############
######################################
from resample import *
from wavToPng import *

inFile = "sa1.wav"
outFile44 = "sa1_out44.wav"
outFile20 = "sa1_out20.wav"
outFile16 = "sa1_out16.wav"
outFile8 = "sa1_out8.wav"
outFile4 = "sa1_out4.wav"
outFile16_from44 = "sa1_out16_from44.wav"

resampleWAV(inFile, outFile44, out_fr=44100, q=1.0)
resampleWAV(inFile, outFile20, out_fr=20000, q=1.0)
resampleWAV(inFile, outFile16, out_fr=16000, q=1.0)
resampleWAV(inFile, outFile16, out_fr=16000, q=1.0)
resampleWAV(inFile, outFile8, out_fr=8000, q=1.0)
resampleWAV(inFile, outFile4, out_fr=4000, q=1.0)
resampleWAV(outFile44, outFile16_from44, out_fr=16000, q=1.0)

wavToPng("sa1.wav")

######################################
############## Fractions #############
######################################
# from fractions import Fraction
# from decimal import Decimal
# print Fraction(Decimal('1.4'))
#
# a= Fraction(Decimal(str(48000/44100.0))).limit_denominator(1000)
# print a.numerator
# print a.denominator
# print(type(a.numerator))
