import librosa
import numpy as np
import scipy.io
from sklearn import preprocessing
import sys

def onehot_matrix(samples_vec, num_classes):
    """
    >>> onehot_matrix(np.array([1, 0, 3]), 4)
    [[ 0.  1.  0.  0.]
     [ 1.  0.  0.  0.]
     [ 0.  0.  0.  1.]]

    >>> onehot_matrix(np.array([2, 2, 0]), 3)
    [[ 0.  0.  1.]
     [ 0.  0.  1.]
     [ 1.  0.  0.]]

    Ref: http://bit.ly/1VSKbuc
    """
    num_samples = samples_vec.shape[0]

    onehot = np.zeros(shape=(num_samples, num_classes))
    onehot[range(0, num_samples), samples_vec] = 1

    return onehot


def wavfile_to_mfccs(wavfile):
    """Returns a matrix of shape (*, 39), since there are 39 MFCCs (deltas
    included for each 20ms segment in the wavfile).
    """
    sampling_rate, frames = scipy.io.wavfile.read(wavfile)

    segment_duration_ms = 20
    n_fft = int((segment_duration_ms / 1000.) * sampling_rate)

    hop_duration_ms = 10
    hop_length = int((hop_duration_ms / 1000.) * sampling_rate)

    mfcc_count = 13

    # n_fft : int > 0 [scalar]:      length of the FFT window
    # hop_length : int > 0 [scalar]: number of samples between successive frames


    mfccs = librosa.feature.mfcc(
            y=frames,
            sr=sampling_rate,
            n_mfcc=mfcc_count,
            hop_length=hop_length,
            n_fft=n_fft
    )
    mfcc_delta = librosa.feature.delta(mfccs)
    #mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    mfccs_and_deltas = np.vstack([mfccs, mfcc_delta])#, mfcc_delta2])
    return mfccs_and_deltas, hop_length, n_fft


def normalize_mean(X, param_path):
    scaler = preprocessing \
        .StandardScaler(with_mean=True, with_std=False) \
        .fit(X)

    np.save(param_path, scaler.mean_)
    X = scaler.transform(X)
    return X


def apply_normalize_mean(X, param_path):
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=False)
    scaler.mean_ = np.load(param_path)

    X = scaler.fit_transform(X)
    return X


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {
        "yes": True, "y": True, "ye": True,
        "no":  False, "n": False
    }
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
