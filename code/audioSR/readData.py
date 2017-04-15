from __future__ import print_function
import os
import numpy as np
from phoneme_set import *
import general_tools
import pdb
import logging

logger_readData = logging.getLogger('evaluate.readData')
logger_readData.setLevel(logging.ERROR)

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    data = cPickle.load(fo)
    fo.close()
    return data

# print overview with results of 1 evaluated file. ! column TARGETS, 1 columnt PREDICTIONS
# both phoneme class numbers and the phoneme itself
def print_results(wav_filenames, inputs, predictions, targets, valid_frames, logger=logger_readData, only_final_accuracy=False):

    try:
        Tfull, Treduced, Tvalid = convertPredictions(targets, valid_frames=valid_frames, outputType = "phonemes")
        Pfull, Preduced, Pvalid = convertPredictions(predictions, valid_frames=valid_frames, outputType = "phonemes")
    except: pdb.set_trace()

    try:
        TfullClass, TreducedClass, TvalidClass = convertPredictions(targets, valid_frames=valid_frames, outputType="classes")
        PfullClass, PreducedClass, PvalidClass = convertPredictions(predictions, valid_frames=valid_frames, outputType="classes")
    except: pdb.set_trace()

    assert len(Tfull) == len(Pfull)

    # print valid predictions, formatted
    logger.debug(wav_filenames)
    logger.debug("    TARGETS \t     PREDICTIONS")
    totalCorrect = 0
    for i in range(len(Tvalid)):
        correct = 0
        if Tvalid[i] == Pvalid[i]: correct = 1
        logger.debug("%s \t %s \t| \t %s \t %s " , Tvalid[i], TvalidClass[i], Pvalid[i], PvalidClass[i])
        totalCorrect += correct
    if not only_final_accuracy: logger.debug("Predicted %d out of %d correctly -> accuracy = %f %%" ,totalCorrect, len(Tvalid),
                                                                     100.0*totalCorrect / len(Tvalid))
    return len(Tvalid), totalCorrect


# print overview of all files that have been evaluated
# count number of correct predictions
def printEvaluation(wav_filenames, inputs, predictions, targets, valid_frames, avgAcc="unknown", indexList=(0,), logger=logger_readData,
                    only_final_accuracy=False):
    # each of these is a list, containing one entry per input wav file
    totalCorrect = 0
    totalSeen = 0
    for index in indexList:
        if not only_final_accuracy: logger.info("\n    RESULTS FOR FILE AT INDEX ", index)

        assert (index >= 0 and index < len(inputs))
        video_wav_filenames = wav_filenames[index]
        video_inputs     = inputs[index]
        video_predictions = predictions[index]
        video_targets      = targets[index]
        video_valid_frames = valid_frames[index]

        # for each video, print the results
        try:n_phonemes, n_correct = print_results(video_wav_filenames, video_inputs, video_predictions, video_targets,
                                                  video_valid_frames, logger, only_final_accuracy=False)
        except:   import pdb;pdb.set_trace()
        totalSeen += n_phonemes
        totalCorrect += n_correct

    #logger.info("expected Avg ACCURACY: -> %s " % (avgAcc))
    logger.info("\n\n%s evaluated files: %d out of %d correct -> accuracy = %f %%" % (len(indexList), totalCorrect, totalSeen,
                                                                    100.0 * totalCorrect / totalSeen))

# pickleFile = os.path.expanduser('~/TCDTIMIT/audioSR/TCDTIMITaudio_resampled/evaluations/volunteers_10M_predictions.pkl')
#pickleFile = os.path.expanduser('~/TCDTIMIT/audioSR/TIMIT/evaluations/volunteers_10M_predictions.pkl')
#[inputs, predictions, targets, valid_frames, avg_Acc] = unpickle(os.path.expanduser(pickleFile))
#printEvaluation(inputs, predictions, targets, valid_frames, avg_Acc)


