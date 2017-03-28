import numpy as np
from itertools import groupby

def regulate(raw_phonemes, max_allowed):
    """
    ~Regulate~ a series of phonemes by removing those which are infrequent, etc.

    Args:
        raw_phonemes: series of phonemes that includes those which are
            inaccurate, etc.

    Returns:
        list of max_allowed elements encapsulating the "correct" phonemes; if
        the list is not filled, right-pad it with zeros.
    """
    # remove single values
    seq = filter_sequence(raw_phonemes)
    reg_phonemes = [k[0] for k in groupby(seq)]

    # does not yet address if sequence is too long (truncate? filter with min_combo = 3?)
    return np.array(pad_list(reg_phonemes, 0, max_allowed)[:max_allowed])

def filter_sequence(seq, min_combo=2):
    # simple way
    combos = [[k, len(list(g))] for k, g in groupby(seq)]
    # one line?: return [x for combo in combos for x in [combo[0]]*combo[1] if combo[1] >= min_combo]
    nseq = []
    for combo in combos:
        if combo[1] >= min_combo:
            # preserve duplication for repeated filtering
            nseq.extend([combo[0]]*combo[1])
    return nseq

def pad_list(seq, pad_val, max_len):
    return seq + [pad_val] * (max_len - len(seq))
