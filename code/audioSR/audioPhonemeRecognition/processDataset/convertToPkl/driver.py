import os
import sys

DEBUG = True

from model import dataset_v2 # , speech2phonemes

if __name__ == '__main__':
    # Obtain information about the commands via:
    # python driver.py --help

    # Suppress stderr from the output
    if not DEBUG:
        null = open(os.devnull, 'wb')
        sys.stderr = null

    one = sys.argv[1]

    if one == 'pkl':
        two = sys.argv[2]  # possibilities: 'all', 'train', 'test'
        print('Generating PKL files...')
        dataset_v2.createPKL(two)

    # elif one == 'train':
    #     print('Training speech2phonemes...')
    #     speech2phonemes.train()
    #
    # elif one == 'test':
    #     print('Testing speech2phonemes...')
    #     speech2phonemes.test()

    else:
        print('Unrecognized action: %s' % one)
