# from http://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output?rq=1
import logging
import os

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

# The background is set with 40 plus the number of the color, and the foreground with 30

# These are the sequences need to get colored ouput
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"


def formatter_message(message, use_color=True):
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message


COLORS = {
    'WARNING':  YELLOW,
    'INFO':     WHITE,
    'DEBUG':    BLUE,
    'CRITICAL': YELLOW,
    'ERROR':    RED
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color=True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)


# Custom logger class with multiple destinations
class ColoredLogger(logging.Logger):
    FORMAT = '%(asctime)s - (%(filename)s:%(lineno)d) | %(message)s'  # "[$BOLD%(name)-5s$RESET][%(levelname)-10s]($BOLD%(filename)s$RESET:%(lineno)d) %(message)s "
    COLOR_FORMAT = formatter_message(FORMAT, True)

    def __init__(self, name):
        logging.Logger.__init__(self, name, logging.WARNING)

        color_formatter = ColoredFormatter(self.COLOR_FORMAT)

        console = logging.StreamHandler()
        console.setFormatter(color_formatter)
        console.setLevel(logging.INFO)

        self.addHandler(console)
        return

    def addFileHandler(self, output_dir='.', log_name="logger.log"):
        fileHandler = logging.FileHandler(os.path.join(output_dir, log_name), "w", encoding=None, delay=True)
        fileHandler.setFormatter(ColoredFormatter(self.COLOR_FORMAT))
        fileHandler.setLevel(logging.DEBUG)
        self.addHandler(fileHandler)
