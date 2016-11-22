from processDatabase import *

# Testing
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG) #http://stackoverflow.com/questions/6579496/using-print-statements-only-to-debug
# levels: debug, info, warning, error and critical.
# logging.debug('A debug message!')
# logging.info('We processed %d records', len(processed_records))

#test=readfile('lipspeaker.mlf')
#print(len(test))
#print(len(test[0]))
#print(test[0])
#video0,phoneme0=processVideoFile(test[0])
#print(video0)
#print(phoneme0)

# get phone list from file, extract phonemes and times, get the frames corresponding to those phonemes
# !!!! Before running this, make sure all the paths to the videos in the MLF file are correct!
processMLF('./lipspeaker_labelfiles.mlf', '/media/matthijs/TOSHIBA EXT/TCDTIMIT/preparedDB')
