import sys, os

DEBUG = True

from model import dataset#, speech2phonemes

if __name__ == '__main__':
    # Obtain information about the commands via:
    # python driver.py --help

    # Suppress stderr from the output
    if not DEBUG:
        null = open(os.devnull,'wb')
        sys.stderr = null

    one = sys.argv[1]

    if one == 'pkl':
        print('Generating PKL files...')
        dataset.Speech2Phonemes().createPKL('all')

    # elif one == 'train':
    #     print('Training speech2phonemes...')
    #     speech2phonemes.train()
    #
    # elif one == 'test':
    #     print('Testing speech2phonemes...')
    #     speech2phonemes.test()

    else:
        print('Unrecognized action: %s' % one)




