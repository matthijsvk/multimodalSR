import os
import sys
import numpy as np
import theano
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD


def read_file_list(files_name):
    """
    convert the to file list
    """
    files_list = []
    fid = open(files_name)
    for line in fid.readlines():
        line = line.strip()
        if len(line) < 1:
            continue
        files_list.append(line)
    fid.close()
    return  files_list


def extract_file_id_list(files_list):
    """
    remove any file extensions
    """
    files_id_list = []
    for file_name in files_list:
        file_id = os.path.basename(os.path.splitext(file_name)[0])
        files_id_list.append(file_id)

    return  files_id_list


def prepare_file_path_list(files_id_list, files_dir, files_extension, 
    new_dir_switch=True):
    if not os.path.exists(files_dir) and new_dir_switch:
        os.makedirs(files_dir)
    files_name_list = []
    for file_id in files_id_list:
        file_name = os.path.join(files_dir, file_id + files_extension)
        files_name_list.append(file_name)

    return  files_name_list


def phoneme_binary(phoneme_list):
    n = len(phoneme_list)
    binary = ['0']*n
    phoneme_bin = {}
    for i in xrange(n):
        ph = phoneme_list[i]
        binary[i] = '1'
        phoneme_bin[ph] = ' '.join(binary)
        binary[i] = '0'

    return phoneme_bin


def lab2binary(in_file_list, ph_bin, out_file_list):
    """
    in_file_list: list of label files
    ph_bin: dictionary for phoneme to binary label
    """
    for i in xrange(len(in_file_list)):
        with open(in_file_list[i],'r') as lab:
            b_label = [ph_bin[l.strip()] for l in lab.readlines()]
        
        with open(out_file_list[i], 'w') as outlab:
            outlab.write('\n'.join(b_label))


def MVN_normalize(in_file_list, mfcc_dim, out_file_list):
    """
    mean and variance normalization
    """
    for i in xrange(len(in_file_list)):
        data = np.fromfile(in_file_list[i], dtype=np.float32, sep=' ', count=-1)
        data.resize(len(data)/mfcc_dim, mfcc_dim)
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data = (data - np.tile(data_mean, (data.shape[0],1)))/np.tile(data_std, (data.shape[0],1))

        np.savetxt(out_file_list[i], data, fmt='%.6f', delimiter=' ')


def whitening():
    """
    whitening normalization
    """


def minmax_normalize(mfcc_file_list, norm_dir):
    """
    minmax normalization
    This normalized the data into the range of [0,1]
    """
    for f in mfcc_file_list:
        data = np.fromfile(f, dtype=float, sep=' ', count=-1)
        data.resize(len(data)/mfcc_dim, mfcc_dim)
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        data = (data - np.tile(data_min, (data.shape[0],1)))/(np.tile(data_max, (data.shape[0],1)) - np.tile(data_min, (data.shape[0],1)))


def load_data(input_file_list, output_file_list, in_dim=39, out_dim=24):
    """
    load partiotion
    """
    for i in xrange(len(input_file_list)):  
        in_data = np.fromfile(input_file_list[i],dtype=np.float32,sep=' ',count=-1)
        out_data = np.fromfile(output_file_list[i],dtype=np.float32,sep=' ',count=-1)
        if i > 0:
            input_data = np.concatenate((input_data,in_data))
            output_data = np.concatenate((output_data, out_data))
        else:
            input_data = in_data
            output_data = out_data
    input_data.resize(len(input_data)/in_dim, in_dim)
    output_data.resize(len(output_data)/out_dim, out_dim)
    np.random.seed(271639)
    np.random.shuffle(input_data)
    np.random.seed(271639)
    np.random.shuffle(output_data)

    return input_data, output_data


def make_shared(data_set, data_name):
    data_set = theano.shared(np.asarray(data_set, dtype=theano.config.floatX), name=data_name, borrow=True)
    return  data_set


def train_mlp(train_x, train_y, hidden_layers, input_dim, output_dim):
    model = Sequential()
    # first layer
    model.add(Dense(hidden_layers[0], input_dim=input_dim, init='uniform'))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.0))
    # hidden layers
    for i in xrange(1,len(hidden_layers)):
        model.add(Dense(hidden_layers[i], input_dim=hidden_layers[i-1],
            init='uniform'))
        model.add(Activation('sigmoid'))
        model.add(Dropout(0.0))
    #output layer
    model.add(Dense(output_dim, input_dim=hidden_layers[-1], init='uniform'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    model.fit(train_x, train_y, nb_epoch=20, batch_size=16)
    #score = model.evaluate(test_x)
    return model


 
## Configuration ##
mfcc_dim = 39
normalization = 'MVN'
project_dir = '/u/97/bollepb1/unix/courses/ASR/project'
data_dir = os.path.join(project_dir, 'data')
mfcc_dir = os.path.join(data_dir, 'mfcc')
labels_dir = os.path.join(data_dir, 'labels')
phoneme_file = os.path.join(data_dir, 'phone.list')
files_scp = os.path.join(data_dir, 'train_files.scp')
files_list = read_file_list(files_scp)
files_id_list = extract_file_id_list(files_list)
no_total_files = len(files_id_list)
no_train_files = int(0.8*no_total_files)
no_valid_files = int(0.1*no_total_files)
#no_test_files = no_total_files - (no_train_files + no_valid_files)

## Flags to run the steps ##
MAKE_LAB = 1
NORM_MFCC = 1
MAKE_DATA = 1
TRAIN_DNN = 1

labels_file_list = prepare_file_path_list(files_id_list,
                                            labels_dir, '.labels', False)

binary_labels_dir = os.path.join(data_dir, 'binary_labels')
binary_labels_file_list = prepare_file_path_list(files_id_list,
                                        binary_labels_dir, '.labels')

mfcc_file_list = prepare_file_path_list(files_id_list,
                                        mfcc_dir, '.mfcc', False)

norm_mfcc_dir = os.path.join(data_dir, 'norm_mfcc')
norm_mfcc_file_list = prepare_file_path_list(files_id_list,
                                        norm_mfcc_dir, '.mfcc')


# convert labels into binary form
if MAKE_LAB:    
    phoneme_list = read_file_list(phoneme_file)
    ph_bin = phoneme_binary(phoneme_list)

    lab2binary(labels_file_list, ph_bin, binary_labels_file_list)


# normalize the mfcc features
if NORM_MFCC:
    MVN_normalize(mfcc_file_list, mfcc_dim, norm_mfcc_file_list)


# prepare data for training
if MAKE_DATA:
    train_y_files = binary_labels_file_list[:no_train_files]
    train_x_files = norm_mfcc_file_list[:no_train_files]

    valid_y_files = binary_labels_file_list[no_train_files:no_train_files+no_valid_files]
    valid_x_files = norm_mfcc_file_list[no_train_files:no_train_files+no_valid_files]

    test_y_files = binary_labels_file_list[no_train_files+no_valid_files:no_total_files]
    test_x_files = norm_mfcc_file_list[no_train_files+no_valid_files:no_total_files]


    train_x, train_y = load_data(train_x_files,train_y_files)
    valid_x, valid_y = load_data(valid_x_files,valid_y_files)
    test_x, test_y = load_data(test_x_files,test_y_files)

    shared_train_x = make_shared(train_x, 'x')
    shared_train_y = make_shared(train_y, 'y')

    shared_valid_x = make_shared(valid_x, 'vx')
    shared_valid_y = make_shared(valid_y, 'vy')

    shared_test_x = make_shared(test_x, 'tx')
    shared_test_y = make_shared(test_y, 'ty')

# train DNNs
if TRAIN_DNN:
    hidden_layers = [100, 100, 100, 100]
    train_model = train_mlp(train_x, train_y, hidden_layers,
                  input_dim=39, 
                  output_dim=24)

