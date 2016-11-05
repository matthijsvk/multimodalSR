This is the repository for the course project of the ELEC-E5510-Speech Recognition. The project title is "Deep Neural Networks for Acoustic Modelling".

Source code is in python and uses theano library.
Easiest way to get the environment setup on windows is to install anaconda (free for students) and then install theano.
Detailed instruction: http://deeplearning.net/software/theano/install.html

Data is put in the data/ folder 

First you need to preprocess the data using preproc script. Extract the .gz data file into a folder 'SRC' in 'data/' folder and provide the 
'SRC' name and a dest folder name (again will be within data/) as input to the preproc script 

evalModel.py  --> use to evaluate accuracy on any data split using the saved model
