import logging
import formatting
import os

logger_RNN = logging.getLogger('RNN')
logger_RNN.setLevel(logging.DEBUG)
FORMAT = '[$BOLD%(filename)s$RESET:%(lineno)d][%(levelname)-5s]: %(message)s '
formatter = logging.Formatter(formatting.formatter_message(FORMAT, False))
formatter2 = logging.Formatter('%(asctime)s - %(name)-5s - %(levelname)-10s - (%(filename)s:%(lineno)d): %(message)s')

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger_RNN.addHandler(ch)

# File logger: see below META VARIABLES
logFile = 'testje.log'
fh = logging.FileHandler(logFile, 'w')  # create new logFile
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger_RNN.addHandler(fh)


logger_RNN.info("This is some info")
logger_RNN.debug("debug")
logger_RNN.critical("Critical")
logger_RNN.info("More info")

from testlogging2 import *
printje(logger=logger_RNN)