from processDatabase import *

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG) #http://stackoverflow.com/questions/6579496/using-print-statements-only-to-debug
# setConsoleLevel(logging.ERROR)
# setFileLevel(logging.INFO)

# get phone list from file, extract phonemes and times, get the frames corresponding to those phonemes

###################################################################################################
# !!!! Before running this, make sure all the paths to the videos in the MLF file are correct !!!!#0
###################################################################################################
startTime = time.clock()

processDatabase('/home/matthijs/TCDTIMIT_test/test.mlf',"/home/matthijs/TCDTIMIT_test/processed", 8) #storeDir requires TCDTIMIT in the name

duration = time.clock() - startTime
print("This took ", duration, " seconds")