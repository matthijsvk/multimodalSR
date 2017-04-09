import logging

logger_RNNtools = logging.getLogger('RNN.tools')
logger_RNNtools.setLevel(logging.DEBUG)


def printje(logger=logger_RNNtools):
    logger.info("This is some info")
    logger.debug("debug")
    logger.critical("Critical")
    logger.info("More info")