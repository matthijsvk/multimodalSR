from __future__ import print_function
import re

# read the phonemes in a list, strip newlines
vocab = [line.rstrip('\n') for line in open('./phonemeList.txt')]

# create dict with phonem and nb of occurences
countLipspeakers = dict((x,0) for x in vocab)

## read MLF files, update the counters (values) of the dict
# lipspeaker file
lines = [line.rstrip('\n') for line in open('../Datasets/TCD-TIMIT/lipspeaker_labelfiles.mlf')]
for line in lines:
    #r = re.compile("|".join(r"\b%s\b" % w for w in vocab))
    for w in re.findall(r"\w+", line):
        if w in countLipspeakers:
            countLipspeakers[w] += 1


# read the volunteer file
countVolunteers = dict((x,0) for x in vocab)

lines = [line.rstrip('\n') for line in open('../Datasets/TCD-TIMIT/volunteer_labelfiles.mlf')]
for line in lines:
    for w in re.findall(r"\w+", line):
        if w in countVolunteers:
            countVolunteers[w] += 1

from collections import Counter
total = dict(Counter(countLipspeakers)+Counter(countVolunteers))


# Do some nice printing
print("For the lipspeakers: ", countLipspeakers)
print ('For the volunteers: ', countVolunteers)
print ('For the whole database: ', total)


# list that contains the key-value tuple pairs, sorted by value
totalSorted = []
for key, value in sorted(total.iteritems(), key=lambda (k,v): (v,k)):
    totalSorted.append((key, value))
    
# print ten first and ten last values
print('Ten least often occuring: ', totalSorted[0:10])
print('Ten most often occuring : ', totalSorted[-10:])

# print total number of phonemes found
totalSum = sum(total.values())
print('Total number of phonemes in database: \t', sum(total.values()) )
print('Avg number of images per phonemes: \t \t', totalSum/39)