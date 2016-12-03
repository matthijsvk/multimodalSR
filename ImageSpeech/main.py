from processDatabase import *

# get phone list from file, extract phonemes and times, get the frames corresponding to those phonemes

###################################################################################################
# !!!! Before running this, make sure all the paths to the videos in the MLF file are correct !!!!#0
###################################################################################################
startTime = time.clock()

processDatabase('./lipspeaker_test.mlf',"/home/matthijs/TCDTIMIT/processedTest", 8) #storeDir requires TCDTIMIT in the name

duration = time.clock() - startTime
print("This took ", duration, " seconds")





