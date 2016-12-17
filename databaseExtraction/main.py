from processDatabase import *

# get phone list from file, extract phonemes and times, get the frames corresponding to those phonemes
# then remove frames without phonemes, extract faces, extract mouths, convert them to grayscale images
# also store compressed (eg 120x120) versions of grayscale faces and mouths

###################################################################################################
# !!!! Before running this, make sure all the paths to the videos in the MLF file are correct !!!!#0
###################################################################################################
startTime = time.clock()

processDatabase('./MLFfiles/lipspeaker_labelfiles.mlf',os.path.expanduser("~/TCDTIMIT/extracted"), 4) #storeDir requires TCDTIMIT in the name
# processDatabase('/home/user/TCDTIMIT_test/test.mlf',os.path.expanduser("~/TCDTIMIT_test/processed"), 4) #storeDir requires TCDTIMIT in the name

duration = time.clock() - startTime
print("This took ", duration, " seconds")





