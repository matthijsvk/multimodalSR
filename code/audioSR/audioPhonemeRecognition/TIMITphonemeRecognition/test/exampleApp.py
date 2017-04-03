import logging
import otherMod2


# ----------------------------------------------------------------------
def main():
    """
    The main entry point of the application
    """
    logger = logging.getLogger("exampleApp")
    logger.setLevel(logging.INFO)
    handler = logging.Handler()
    logger.addHandler(handler)
    print('1')
    logger.info("Program started")
    print('b')
    result = otherMod2.add(7, 8)
    print('2')
    logger.info('%s', result)
    logger.info("Done!")


if __name__ == "__main__":
    main()