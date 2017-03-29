print('\n\n * Imporing Libaries')
import os
import time;

program_start_time = time.time()

from general_tools import *
from RNN_tools import *

##### SCRIPT META VARIABLES #####
print(' * Setting up ...')

comput_confusion = False
# TODO: ATM this is not implemented


dataRootPath = "/home/matthijs/TCDTIMIT/TIMIT/fixedWav/TIMIT/"
train_source_path = os.path.join(dataRootPath, 'TRAIN')
test_source_path = os.path.join(dataRootPath, 'TEST')
data_path = os.path.join(dataRootPath, 'std_preprocess_26_ch.pkl')

output_path = "/home/matthijs/TCDTIMIT/TIMIT/fixedWav/results"
model_load = os.path.join(output_path, 'best_model.npz')
model_save = os.path.join(output_path, 'best_model')

##### SCRIPT VARIABLES #####
num_epochs = 20

NUM_OUTPUT_UNITS = 61
N_HIDDEN = 275

LEARNING_RATE = 1e-5
MOMENTUM = 0.9
WEIGHT_INIT = 0.1
batch_size = 1

INPUT_SIZE=26  #num of features to use

##### IMPORTIN DATA #####
print('\tdata source: ' + data_path)
print('\tmodel target: ' + model_save + '.npz')
dataset = load_dataset(data_path)
X_train, y_train, X_val, y_val, X_test, y_test = dataset

##### BUIDING MODEL #####
print(' * Building network ...')
RNN_network = NeuralNetwork('RNN', dataset, batch_size=batch_size, num_features=INPUT_SIZE, n_hidden=N_HIDDEN,
                            num_output_units=NUM_OUTPUT_UNITS, seed=int(time.time()), debug=False)
RNN_network.load_model(model_load)

##### BUIDING FUNCTION #####
print(" * Compiling functions ...")
RNN_network.build_functions(LEARNING_RATE=LEARNING_RATE, MOMENTUM=MOMENTUM, debug=False)

##### TRAINING #####
print(" * Training ...")
RNN_network.train(dataset, model_save, num_epochs=num_epochs,
                  batch_size=batch_size, compute_confusion=False, debug=False)

print()
print(" * Done")
print()
print('Total time: {:.3f}'.format(time.time() - program_start_time))
