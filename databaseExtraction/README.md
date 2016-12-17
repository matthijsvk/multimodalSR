Hi there.  
This file explains some things about the scripts that process the TCDTIMIT database.  

The goal is to extract from mp4 videos and mlf phoneme label files the video frames that correspond to the times of the phonemes in the label file.  
This is desired to be able to train a NN on images of people pronouncing a phoneme, and do lipreading.

What you'll need:  
- create account on https://sigmedia.tcd.ie/TCDTIMIT/ in order to be able to download the database  
- linux system with bash  
- python (I used 2.7.12)  
- dlib (use Anaconda for easy installation: 'conda install dlib')  
- unzip  

The most important files are:  

1. `downloadTCDTIMIT.sh`: 
    - a bash script that downloads the database in the same directory as where the script is executed. All files that it downloads are in `downloadLocations.txt`  
    - You need to request an account and login to the sigmedia website (https://sigmedia.tcd.ie/TCDTIMIT/) first. More instructions inside the script.  
                        
2. `lipspeaker_labelfiles.mlf` and `volunteer_labelfiles.mlf`: 
    - these files are used as input in the main.py file. 
    - You can use the `lipspeaker_test.mlp` for a (much) smaller version of the database, to try it out. You'll have to extract the zip files first, though.  

3. `main.py`: 
    - use this to specify the main MLF file, which contains paths of the videos to be processed, and start-and stop times for each phoneme in the videos.  You also set the target directory and the number of threads to use.  
    - In order to process the database, you need to change all the paths in this file to the paths where you downloaded the database.  
            
4. `processDatabase.py`:
      - this file contains the main functions that process the downloaded database. It works multithreadedly.  
      - extracted images and files will be saved to storageLocation (set in main.py), but further the directory structure of the database will be kept (so an extracted frame is stored in storageLocation/lipreaders/Lipspkr1/sa1/extractedFrame.jpg)  
      - it works in several steps:    
           1. preprocess the MLF file: extract video paths, phonemes and their times, and the corresponding frames in the video.    
           1. extract all frames from the video, cropping the video to some box roughly corresponding to the face (to make further processing easier)     
           1. remove unneccessary frames, using information from (1)  
           1. The phonemes and corresponding frame numbers are stored in a text file in each video's folder, named 'speaker_video_phonemeTimings.txt'    
           1. extract faces and mouths, storing them in 'faces' and 'mouths' folders    
           1. resize the mouth images, to make them all the same size  
5. `helpFunctions.py`: contains lots of functions that are used in processDatabase.py, in order to keep that file somewhat clean    
6. other files: 
    1. `EGillenThesis.pdf`:   the masters thesis of Eoin Gillen, Trinity College Dublin, who helped create the database. This contains lots of info about how the database was made.    
    1. `phonemeList.txt`:     contains a list of all possible phonemes (39). This is somewhat simplified from the full english phoneme set, see EGillenThesis.pdf. These are also in CMU_..._phones.txt  
    1. `CMU_Pronunciation_Dictionary`: can be used to map phonemes to words or the other way around. See for example `https://github.com/jimkang/word-phoneme-map`  
    1. `unzipStructure.py`:   recursively searches a directory tree for zip files, and extracts them in the same place. You can choose to delete or keep the zip files after extraction.  
    1. `countNbPhonemes.py`:  counts the number of times the database contains each phoneme. Useful for estimating how well you'll be able to train.  
    1. `fileDirOps.py`:       provides some renaming functions, as well as a function to recursively search for and remove directories with certain names (for example 'faces' directories)  
    1. `dlib.so`:             I needed to place this file in this folder in order for the face and mouth extraction to work.  
    1. `shape_predictor_68_face_landmarks.dat`: used by dlib to detect 68 facial landmarks, wihch are used to the mouth region. See `dlibLandmarks.png` for a visualization of the landmark locations.  
