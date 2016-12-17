Hi there.  
This file explains how to use these scripts in order to process the TCDTIMIT database.

First of all, I want to thank the people at the Sigmedia department of the Trinity College Dublin for creating this database.
Harte, N.; Gillen, E., "TCD-TIMIT: An Audio-Visual Corpus of Continuous Speech," Multimedia, IEEE Transactions on , vol.17, no.5, pp.603,615, May 2015 doi: 10.1109/TMM.2015.2407694

The goal is to extract from mp4 videos and mlf phoneme label files the video frames that correspond to the phonemes in the label file.

What you'll need:  
- an account on https://sigmedia.tcd.ie/TCDTIMIT/ in order to be able to download the database
- linux system with bash  
- python (I used 2.7.12). I recommend using Anaconda
- dlib (image processing library). Anaconda installation command: 'conda install dlib'
- unzip  

The most important folders and files are:

1. the folder downloadTCDTIMIT: `downloadTCDTIMIT.sh` and unzipStructure.py:
    - a bash script that downloads the database in the same directory as where the script is executed. All files that it downloads are in `downloadLocations.txt`  
    - You need to request an account and login to the sigmedia website (https://sigmedia.tcd.ie/TCDTIMIT/) first. More instructions inside the script.  
    -  Then you can use the python script to unzip all of the zip files.

2. in the folder MLFfiles: `lipspeaker_labelfiles.mlf` and `volunteer_labelfiles.mlf`:
    - these files are used as input in the main.py file. 
    - You can use the `lipspeaker_test.mlf` for a (much) smaller version of the database, to try it out. You'll have to extract the zip files first, though.

3. `main.py`: 
    - use this to specify the main MLF file, which contains paths of the videos to be processed, and start-and stop times for each phoneme in the videos.  You also set the target directory and the number of threads to use.  
    - In order to process the database, you need to change all the paths in this file to the paths where you downloaded the database.  
            
4. `processDatabase.py`:
      - this file contains the main functions that process the downloaded database. It works multithreadedly.  
      - extracted images and files will be saved to storageLocation (set in main.py), but further the directory structure of the database will be kept (so an extracted frame is stored in storageLocation/lipreaders/Lipspkr1/sa1/extractedFrame.jpg)  
      - it works in several steps:    
           1. preprocess the MLF file: extract video paths, phonemes and their times, and the corresponding frames in the video.    
           1. extract all frames from the video, cropping the video to some box roughly corresponding to the face (to make further processing easier)
           1. write phoneme-frame information to both a txt and a mat file
           1. remove unneccessary frames
           1. extract faces and mouths, storing them in 'faces' and 'mouths' folders
           1. convert images to grayscale
           1. resize the mouth images, to make them all the same size
      - there are several parameters you can change for your application
5. `helpFunctions.py`: contains implementations of functions that are used in processDatabase.py

6. `fileDirOps.py`:  This was used to filter the extracted database files for grayscale resized mouth images and the phoneme-frame txt files. The phoneme corresponding to each image was added to the image name. The files were then copied to another directory, and pickled for later usage in neural networks.

7. other files:
    1. `EGillenThesis.pdf`:   the masters thesis of Eoin Gillen, Trinity College Dublin, who helped create the database. This contains lots of info about how the database was made.    
    1. `phonemeList.txt`:     contains a list of all possible phonemes (39). This is somewhat simplified from the full english phoneme set, see EGillenThesis.pdf. These are also in CMU_..._phones.txt  
    1. `CMU_Pronunciation_Dictionary`: can be used to map phonemes to words or the other way around. See for example `https://github.com/jimkang/word-phoneme-map`  
    1. `countNbPhonemes.py`:  counts the number of times the database contains each phoneme. Useful for estimating how well you'll be able to train.  
    1. `dlib.so`:             I needed to place this file in this folder in order for the face and mouth extraction to work.  
    1. `shape_predictor_68_face_landmarks.dat`: used by dlib to detect 68 facial landmarks, wihch are used to the mouth region. See `dlibLandmarks.png` for a visualization of the landmark locations.  
    
    
In short: in order to use the database, do this:
1. download and extract it (folder downloadTCDTIMIT)
2. update the MLF files for lipspeakers and volunteers to point to wherever you downloaded and extracted the database
3. change the paths in main.py, i suggest first extracting the lipspeakers (you can use the test MLF as well)
4. wait
5. if you want to do further extraction, you could use fileDirOps.py to get specific files/folders and/or pickle them
6 done!

If you have any questions, you can reach me under:
matthijsvankeirsbilck@gmail.com