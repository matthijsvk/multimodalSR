from . import phonemes2text, regulator, speech2phonemes


def predict(recording):
    """
    Run the user input through the entire model. Take a series of MFCCs from the
    audio recording and output the predicted words (there number of words
    depends on the length of the provided input).

    1. speech2phonemes
    2. regulator
    3. phonemes2text

    Args:
        recording: matrix of shape (*, 39) <-- see utils.wavfile_to_mfccs()

    Returns:
        list of predicted words
    """
    # recording.shape = (mfcc_series, 39)

    # Get the phonemes matching each MFCC group (of 20ms)
    # phonemes.shape = (mfcc_series, 1)
    phonemes = speech2phonemes.predict(recording.transpose())
    nphonemes = dataset.Speech2Phonemes()._wavfile_apply_normalizer(phonemes)[0]

    # Group the phonemes into a word
    rphonemes = regulator.regulate(nphonemes, max_allowed=30).reshape((1, 30))
    word = phonemes2text.predict(rphonemes)

    return word
