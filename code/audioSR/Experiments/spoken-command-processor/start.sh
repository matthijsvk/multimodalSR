#!/bin/bash

# The root of the project is the root of this script
# http://stackoverflow.com/a/246128/2708484
export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $PROJECT_ROOT

#################################
# Config. environment variables #
#################################

# Path to the trained model's parameters
MODEL_PARAMETERS="$PROJECT_ROOT/volumes/model"
export MODEL_PARAMETERS
echo $MODEL_PARAMETERS

# Path to the list of phones in the data
PHONE_LIST_PATH="$PROJECT_ROOT/volumes/config/timit_phones.txt"
export PHONE_LIST_PATH

# Path to the list of words in the data
WORD_LIST_PATH="$PROJECT_ROOT/volumes/config/timit_words.txt"
export WORD_LIST_PATH

# Paths to training and testing portions of the dataset
TIMIT_TRAINING_PATH="/home/matthijs/TCDTIMIT/TIMIT/fixed/TIMIT/TRAIN/"
export TIMIT_TRAINING_PATH
TIMIT_TESTING_PATH="/home/matthijs/TCDTIMIT/TIMIT/fixed/TIMIT/TEST/"
export TIMIT_TESTING_PATH

if [ -z "$TIMIT_TRAINING_PATH" ] || [ -z "$TIMIT_TESTING_PATH" ]; then
  echo "Set env. vars: TIMIT_TRAINING_PATH and TIMIT_TESTING_PATH."
  return;
fi

echo $TIMIT_TRAINING_PATH
echo $TIMIT_TESTING_PATH

# Path to a temporary wavfile that is created to transfer data from the
# recorder to the model
TMP_RECORDING="$PROJECT_ROOT/tmp_recording.wav"
export TMP_RECORDING

######################################
# Add relevant modules to PYTHONPATH #
######################################
PYTHONPATH="$PYTHONPATH:$PROJECT_ROOT"
export PYTHONPATH

for dir in $PROJECT_ROOT/*; do
  PYTHONPATH="$PYTHONPATH:$dir"
  export PYTHONPATH
done


echo $PROJECT_ROOT
