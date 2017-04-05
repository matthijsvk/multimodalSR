# using the 39 phone set proposed in (Lee & Hon, 1989)
# Table 3. Mapping from 61 classes to 39 classes, as proposed by Lee and Hon, (Lee & Hon,
# 1989). The phones in the left column are folded into the labels of the right column. The
# remaining phones are left intact.

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
