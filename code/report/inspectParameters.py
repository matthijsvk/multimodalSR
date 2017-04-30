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

def load_model(model_path, logger=logger_inspectParameters):
    logger.info("Loading stored model...")

    # restore network weights
    with np.load(model_path) as f:
        all_params = [f['arr_%d' % i] for i in range(len(f.files))][0]

    logger.info("number of layers: %s", len(all_params))

    for i in range(len(all_params)):
        layer_params = all_params[i]
        logger.info("layer %s.shape: %s", i, layer_params.shape)

    import pdb;pdb.set_trace()

model_path = '/home/matthijs/TCDTIMIT/audioSR/combined/results/BEST/2_LSTMLayer64_64_nbMFCC39_bidirectional_combined.npz'
load_model(model_path=model_path)