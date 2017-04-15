from __future__ import print_function
import os
import numpy as np

def getPhonemeNumberMap(phonemeMap="./phonemeLabelConversion.txt"):
    phonemeNumberMap = {}
    with open(phonemeMap) as inf:
        for line in inf:
            parts = line.split()  # split line into parts
            if len(parts) > 1:  # if at least 2 parts/columns
                phonemeNumberMap[str(parts[0])] = parts[1]  # part0= frame, part1 = phoneme
                phonemeNumberMap[str(parts[1])] = parts[0]
    return phonemeNumberMap


phonemeNumberMap = getPhonemeNumberMap()
phoneme = 'aa'
classNumber = phonemeNumberMap[phoneme]
print(classNumber, type(classNumber))


classNumber = np.array([classNumber]).astype('int32')
print(classNumber, type(classNumber[0]))
