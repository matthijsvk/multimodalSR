import logging
import formatting

logger_inspectParameters = logging.getLogger('inspectParameters')
logger_inspectParameters.setLevel(logging.DEBUG)
FORMAT = '[$BOLD%(filename)s$RESET:%(lineno)d][%(levelname)-5s]: %(message)s '
formatter = logging.Formatter(formatting.formatter_message(FORMAT, False))

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger_inspectParameters.addHandler(ch)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib

matplotlib.rcParams.update({'font.size': 22})


def load_model(model_path, logger=logger_inspectParameters):
    logger.info("Loading stored model...")

    # restore network weights
    with np.load(model_path) as f:
        all_params = [f['arr_%d' % i] for i in range(len(f.files))][0]

    logger.info("number of layers: %s", len(all_params))

    for i in range(len(all_params)):
        layer_params = all_params[i]
        logger.info("layer %s.shape: %s", i, layer_params.shape)

    return all_params
# model_path = '/home/matthijs/TCDTIMIT/audioSR/combined/results/BEST/2_LSTMLayer64_64_nbMFCC39_bidirectional_combined.npz'
# load_model(model_path=model_path)

# lipreading dense, audio raw features
model_path = '/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/results/CNN_LSTM/lipspeakers/' \
             'RNN__2_LSTMLayer256_256_nbMFCC39_bidirectional' \
             '__CNN_google_dense_lipRNN_256_256_RNNfeaturesDense' \
             '__FC_512_512_512__TCDTIMIT_lipspeakers.npz'
# lipreading
model_path ='/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/results/CNN_LSTM/lipspeakers/' \
            'RNN__2_LSTMLayer256_256_nbMFCC39_bidirectional' \
            '__CNN_google_dense_lipRNN_256_256' \
            '__FC_512_512_512__TCDTIMIT_lipspeakers.npz'
paramsArray = load_model(model_path=model_path)

# CNN features : layer 18
# CNN-LSTM features: layer 29
# audio features:
# combined features weights: layers 122 (768x 512)
combined_features = paramsArray[122]

lip_features = combined_features[:256]
lip_features = np.abs(lip_features.flatten())

mean_lip = np.mean(lip_features)
median_lip = np.median(lip_features)
rms_lip = np.sqrt(np.mean(np.square(lip_features)))

print("lipreading mean: ", mean_lip)
print("lipreading median: ", median_lip)
print("lipreading rms: ", rms_lip)

audio_features = combined_features[256:]
audio_features = np.abs(audio_features.flatten())
mean_audio = np.mean(audio_features)
median_audio = np.median(audio_features)
rms_audio = np.sqrt(np.mean(np.square(audio_features)))

print("audio mean: ", mean_audio)
print("audio median: ", median_audio)
print("audio rms: ", rms_audio)

lipreading mean:  0.0469951
lipreading median:  0.041638106
lipreading rms:  0.057422262
audio mean:  0.04470453
audio media:  0.039826244
audio rms:  0.054539148

showFigs = True
if showFigs:
    fig = figure()
    ax = fig.add_subplot(111)
    ax.hist(combined_features.flatten(), bins='auto')  # plt.hist passes it's arguments to np.histogram
    #ax.boxplot(combined_features.flatten())  # , cmap='binary')
    ax.set_xlabel("weight size")
    ax.set_ylabel("number of weights")
    plt.show()

    fig = figure()
    ax = fig.add_subplot(111)
    ax.hist(lip_features.flatten(), bins='auto')  # , cmap='binary')
    ax.set_title("FC weight size for Lipreading features ")
    ax.set_xlabel("weight size")
    ax.set_ylabel("number of weights")
    plt.show()

    fig = figure()
    ax = fig.add_subplot(111)
    ax.hist(audio_features.flatten(), bins='auto')  # , cmap='binary')
    ax.set_title("FC weight size for Audio features ")
    ax.set_xlabel("weight size")
    ax.set_ylabel("number of weights")

    plt.show()






