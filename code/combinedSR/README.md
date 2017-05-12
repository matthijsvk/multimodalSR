### Preprocessing 

1. use scripts from TCDTIMITprocessing (main.py) to generate data per video:    
    - valid images and image .phn files  -> [TCDTIMITprocessing](https://github.com/matthijsvk/TCDTIMITprocessing), main.py
    - wavs and phoneme philes ->    
        1. [TCDTIMITprocessing](https://github.com/matthijsvk/TCDTIMITprocessing), extractTCDTIMITaudio.py (get wavs and phonemes)    
        2. [audioSR](https://github.com/matthijsvk/multimodalSR/tree/master/code/audioSR), fixDataset/transform.py (fix headers, resample to 16kHz, can also replace phonemes by different set)    

1. once you have the videos and wavs processed:
    1. copy mouth_120x120 images from `processed` dir to `database` dir using `combined/fileDirOps`.  
       It will rename the images to the format `videoname_frame_phoneme.jpg`  
    1. copy the fixed wavs and wav phoneme files  from step (1) to merge wav and image directories

1. Now run datasetToPkl.py, which generates a pickle file for each speaker. It contains a dictionary with 5 keys. Each key's corresponding value is a list of length (nbVideos), and each element of the list is an array with the corresponding data:    
    1. 'images':  shape (nbVideos, nbValidFrames, 14400)  
     => (per valid frame of the video, a 120x120 image, flattened)
    2. 'mfccs':  shape (nbVideos, nbAudioFrames, nbMFCCs)  
     => nbAudioFrames is variable per video. nbMFCCS is by default 39 (13MFCCS + 1st and 2nd derivatives)  
    3. 'audioLabels': shape (nbVideos, nbAudioFrames)  
     => label is converted from string to the corresponding class number (see phoneme_set.py)  
    4. 'validLabels': shape (nbVideos, nbValidFrames)  
    5. 'validAudioFrames': shape (nbVideos, nbValidFrames)   
     => array of the valid frames of this video (audio frames in middle of phoneme intervals, where we want to do our predictions.  

### Training and evaluating the network
0. if you load a pretrained CNN network from lipreading, make sure to use the same train/test/validation sets for pretraining the lipreading as the combined set.
Use lipreading/datasetToPkl_fromCombined.py to re-flatten the image files so they work with the lipreading scripts.  
1. Now you can configure the network in combinedNN.py, and run it to train on the dataset.
2. You can specify lots of parameters for your network, see the examples in that `combinedNN.py`. You can enter a list of many different networks that will then be automatically trained.  
The results will be stored in the appropriate locations so they can be loaded/evaluated easily.  
3. If a network already exists, it is only evaluated and not trained by default. Override this by setting forceTraining to True. There are some other parameters as well. 
 
You can evaluate your networks faster by using only the lipspeakers instead of the whole dataset.
See `datasetToPkl_lipspeakers.py`.   
This preprocess the lipspeakers only, and stores all dataa in 3 files 1 train, 1 validation, 1 test
These files can be loaded into memory once, and stay there so you don't need to wait for loading/unloading data from memory
like with the full dataset or the volunteers. This can help you get results faster.
To use this, set dataType to "lipspeakers" in combinedNN.py.  
     