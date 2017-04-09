# using the 39 phone set proposed in (Lee & Hon, 1989)
# Table 3. Mapping from 61 classes to 39 classes, as proposed by Lee and Hon, (Lee & Hon,
# 1989). The phones in the left column are folded into the labels of the right column. The
# remaining phones are left intact.
import logging
logger_phonemeSet = logging.getLogger('phonemeSet')
logger_phonemeSet.setLevel(logging.ERROR)

phoneme_set_61_39 = {
    'ao':   'aa',  # 1
    'ax':   'ah',  # 2
    'ax-h': 'ah',
    'axr':  'er',  # 3
    'hv':   'hh',  # 4
    'ix':   'ih',  # 5
    'el':   'l',  # 6
    'em':   'm',  # 6
    'en':   'n',  # 7
    'nx':   'n',
    'eng':  'ng',  # 8
    'zh':   'sh',  # 9
    "ux":   "uw",  # 10
    "pcl":  "sil",  # 11
    "tcl":  "sil",
    "kcl":  "sil",
    "qcl":  "sil",
    "bcl":  "sil",
    "dcl":  "sil",
    "gcl":  "sil",
    "h#":   "sil",
    "#h":   "sil",
    "pau":  "sil",
    "epi":  "sil",
    "q":    "sil",
}

# from https://www.researchgate.net/publication/275055833_TCD-TIMIT_An_audio-visual_corpus_of_continuous_speech
phoneme_set_39_list = [
    'iy', 'ih', 'eh', 'ae', 'ah', 'uw', 'uh', 'aa', 'ey', 'ay', 'oy', 'aw', 'ow',
    'l', 'r', 'y', 'w', 'er', 'm', 'n', 'ng', 'ch', 'jh', 'dh', 'b', 'd', 'dx',
    'g', 'p', 't', 'k', 'z', 'v', 'f', 'th', 's', 'sh', 'hh', 'sil'
]
values = [i for i in range(0, len(phoneme_set_39_list))]
phoneme_set_39 = dict(zip(phoneme_set_39_list, values))
classToPhoneme39 = dict((v, k) for k, v in phoneme_set_39.iteritems())

# from http://www.intechopen.com/books/speech-technologies/phoneme-recognition-on-the-timit-database, page 5
phoneme_set_61_list = [
    'iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah', 'ao', 'oy', 'ow', 'uh', 'uw', 'ux', 'er', 'ax', 'ix', 'axr',
    'ax-h', 'jh',
    'ch', 'b', 'd', 'g', 'p', 't', 'k', 'dx', 's', 'sh', 'z', 'zh', 'f', 'th', 'v', 'dh', 'm', 'n', 'ng', 'em', 'nx',
    'en', 'eng', 'l', 'r', 'w', 'y', 'hh', 'hv', 'el', 'bcl', 'dcl', 'gcl', 'pcl', 'tcl', 'kcl', 'q', 'pau', 'epi',
    'h#',
]
values = [i for i in range(0, len(phoneme_set_61_list))]
phoneme_set_61 = dict(zip(phoneme_set_61_list, values))


def convertPredictions(predictions, phoneme_list=classToPhoneme39, valid_frames=None, outputType="phonemes"):
    # b is straight conversion to phoneme chars
    predictedPhonemes = [phoneme_list[predictedClass] for predictedClass in predictions]

    # c is reduced set of b: duplicates following each other are removed until only 1 is left
    reducedPhonemes = []
    for j in range(len(predictedPhonemes) - 1):
        if predictedPhonemes[j] != predictedPhonemes[j + 1]:
            reducedPhonemes.append(predictedPhonemes[j])

    # get only the outputs for valid phrames
    validPredictions = [predictedPhonemes[frame] for frame in valid_frames]

    # return class outputs
    if outputType!= "phonemes":
        predictedPhonemes = [phoneme_set_39[phoneme] for phoneme in predictedPhonemes]
        reducedPhonemes = [phoneme_set_39[phoneme] for phoneme in reducedPhonemes]
        validPredictions = [phoneme_set_39[phoneme] for phoneme in validPredictions]

    return predictedPhonemes, reducedPhonemes, validPredictions


