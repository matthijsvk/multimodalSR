from __future__ import print_function
import os
import numpy as np
from phoneme_set import *
import general_tools
import pdb

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    data = cPickle.load(fo)
    fo.close()
    return data

def print_results(input, prediction, target, valid_frame):
    # print("input -> len: %s  | values: %s " % (len(input), input))
    # print("Pred:  -> len: %s  | values: %s " % (len(prediction), prediction))
    # print("Target: -> len: %s  | values: %s " % (len(target), target))
    # # print(predictions[0][0] - targets[0])
    # print("validFrames: -> len: % s | values: % s" % (len(valid_frame), valid_frame))

    Tfull, Treduced, Tvalid = convertPredictions(target, valid_frames=valid_frame, outputType = "phonemes")
    Pfull, Preduced, Pvalid = convertPredictions(prediction, valid_frames=valid_frame, outputType = "phonemes")

    TfullClass, TreducedClass, TvalidClass = convertPredictions(target, valid_frames=valid_frame, outputType="classes")
    PfullClass, PreducedClass, PvalidClass = convertPredictions(prediction, valid_frames=valid_frame, outputType="classes")

    try:    assert len(Tfull) == len(Pfull)
    except: pdb.set_trace()

    # print valid predictions, formatted
    print("    TARGETS \t     PREDICTIONS")
    for i in range(len(Tvalid)):
        print("%s \t %s \t| \t %s \t %s" %(Tvalid[i], TvalidClass[i], Pvalid[i], PvalidClass[i]))


#pickleFile = os.path.expanduser('~/TCDTIMIT/audioSR/TCDTIMITaudio_resampled/evaluations/volunteers_10M_predictions.pkl')
def printEvaluation(inputs, predictions, targets, valid_frames, avgAcc="unknown", indexList=(0,)):
    # each of these is a list, containing one entry per input wav file
    print("Avg ACCURACY: -> %s " % (avgAcc))

    for index in indexList:
        print("\n    RESULTS FOR FILE AT INDEX ", index)
        assert (index >= 0 and index < len(inputs))
        input       = inputs[index]
        prediction  = predictions[index]
        target      = targets[index]
        valid_frame = valid_frames[index]

        print_results(input, prediction, target, valid_frame)


pickleFile = os.path.expanduser('~/TCDTIMIT/audioSR/TIMIT/evaluations/volunteers_10M_predictions.pkl')
[inputs, predictions, targets, valid_frames, avg_Acc] = unpickle(os.path.expanduser(pickleFile))
printEvaluation(inputs, predictions, targets, valid_frames, avg_Acc)


