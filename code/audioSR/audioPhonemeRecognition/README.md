1. Process Dataset  
    1. Fix dataset source files
        - TIMIT stores wavs in weird NIST format, we need to convert to normal WAV
        - TIMIT PHN files need to be mapped from 61 to 39 phonemes (mapping: see phoneme_set.py)
            - use `processDataset/fixDataset/transform.py` for this:   
        `python processDataset/fixDataset/transform.py phonemes -i /home/matthijs/TCDTIMIT/TIMIT/original/TIMIT/ -o /home/matthijs/TCDTIMIT/TIMIT/tmp`
        
        - You can generate an MLF (Maste Label File) file containing all phoneme frame and label info:  
             `python processDataset/fixDataset/createMLF.py /home/matthijs/TCDTIMIT/TIMIT/tmp/`
        
        - Specific for TCDTIMIT:
            - Resample audio WAV files so they are all 16kHz (not 48kHz as in TCDTIMIT)  
              `processDataset/fixDataset/resample.py TCDTIMIT_dir/ targetDir/`
              
            - Audio label files should use sample frames, not time + extract seperate .PHN files from MLF  
            -> Get TCDTIMIT .PHN files  
                `python getPhnFiles.py "./MLFfiles/lipspeaker_labelfiles.mlf" ~/TCDTIMIT/TCDTIMITaudio`
     
     2. Use fixed dataset to generate .pkl files
        - load in data + phonemes
        - convert wav to mfcc
        - label to classnumber
        - label samplenr to frame window
        - mean (+ std dev) normalization
        - one hot encoding of targets
        


2. Use .pkl files as input to the network