import os
import pickle
import sys

import librosa
import numpy as np
import scipy
import theano

from audioPhonemeRecognition.processDataset.fixDataset.old import fixWavs

TIMIT_original_dir = '/home/matthijs/TCDTIMIT/TIMIT/original'
TIMIT_fixed_dir = '/home/matthijs/TCDTIMIT/TIMIT/fixed'

def get_data(rootdir = TIMIT_fixed_dir):    
    inputs = []
    targets = []
    alphabet = {}

    # count number of files for showing progress.
    wavCounter = 0
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith('.wav'):
                wavCounter += 1
    print "There are ", wavCounter, " files to be processed"
    from progress_bar import show_progress
    processed = 0

    for dir_path, sub_dirs, files in os.walk(rootdir):
        for file in files:
            if (os.path.join(dir_path, file)).endswith('.wav'):

                ## Get the data itself: inputs and targets
                #--------------------------
                wav_file_name = os.path.join(dir_path, file)

                # from https://github.com/dtjchen/spoken-command-processor/blob/master/model/utils.py
                sampling_rate, frames = scipy.io.wavfile.read(wav_file_name)

                segment_duration_ms = 20
                n_fft = int((segment_duration_ms / 1000.) * sampling_rate)

                hop_duration_ms = 10
                hop_length = int((hop_duration_ms / 1000.) * sampling_rate)

                mfcc_count = 13

                mfccs = librosa.feature.mfcc(
                        y=frames,
                        sr=sampling_rate,
                        n_mfcc=mfcc_count,
                        hop_length=hop_length,
                        n_fft=n_fft
                )
                mfcc_delta = librosa.feature.delta(mfccs)
                mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
                #full_input = np.vstack([mfccs, mfcc_delta, mfcc_delta2])
                full_input = np.concatenate((mfccs, mfcc_delta, mfcc_delta2), axis=1)

                inputs.append(np.asarray(full_input, dtype=theano.config.floatX))

                text_file_name = wav_file_name[:-4] + '.txt'
                target_data_file = open(text_file_name)
                target_data = str(target_data_file.read()).lower().translate(None, '!:,".;?')
                target_data = target_data[8:-1] #No '.' in lexfree dictionary
                targets.append(target_data)

                ## Get alphabet
                # ------------------------
                transcription_filename = os.path.join(dir_path, file)[:-4] + '.txt'
                transcription_file = open(transcription_filename, 'r')
                transcription = str(transcription_file.read()).lower().translate(None, '!:,".;?')
                transcription = transcription[8:-1]

                # count number of occurences of each character
                for char in transcription:
                    if not char in alphabet:
                        alphabet.update({char: 1})
                    else:
                        alphabet[char] += 1

                processed += 1
                if (processed % 100 == 0):
                    show_progress(float(processed) / wavCounter)
                    print " | Read", processed, "files out of", wavCounter

    print 'TIMIT Alphabet:\n', alphabet
    alphabet_filename = 'TIMIT_Alphabet.pkl'
    with open(alphabet_filename, 'wb') as f:
        pickle.dump(alphabet, f, protocol=2)

    return inputs, targets, alphabet

# convert the 61 phonemes from TIMIT to the reduced set of 39 phonemes -> preprocessing: substitute_phones.py

#################################################################
##### TODO read in phoneme data: see SpokenCommandProcessor/model/dataset.py and SCP/model/utils.py
###############################################################

def get_TIMIT_targets_one_hot(inputs, targets, alphabet):
    list_of_alphabets = [key for key in alphabet]
    list_of_alphabets.sort()
    # print list_of_alphabets

    num_targets = len(list_of_alphabets)
    # print len(targets[0])
    # targets_as_alphabet_indices = [[seq.index(char) for char in seq] for seq in targets]
    one_hot_targets = [[np.zeros((num_targets)) for char in example] for example in targets]
    # print len(one_hot_targets[0]), one_hot_targets[0]#, len(one_hot_targets[0][0][0])
    for example_num in range(len(targets)):
        for char_num in range(len(targets[example_num])):
            # print targets[example_num][char_num]
            # print list_of_alphabets.index(targets[example_num][char_num])
            one_hot_targets[example_num][char_num][list_of_alphabets.index(targets[example_num][char_num])]=1
    return one_hot_targets

def get_TIMIT_targets_as_alphabet_indices(inputs, targets, alphabet):
    list_of_alphabets = [key for key in alphabet]
    list_of_alphabets.sort()
    print('list of alphabets: {}'.format(list_of_alphabets))
    print len(list_of_alphabets)
    #print list_of_alphabets.index(22)
    print targets[0]
    targets_as_alphabet_indices = [[list_of_alphabets.index(char) for char in target] for target in targets]
    print "Example target and alphabet indices: "
    print 'target = {} \n alphabet indices = {}'.format(targets[0], targets_as_alphabet_indices[0])

    return targets_as_alphabet_indices

def index2char_TIMIT(input_index_seq = None, TIMIT_pkl_file = os.path.join(os.getcwd(),'TIMIT_data_prepared_for_CTC.pkl')):
    with open(TIMIT_pkl_file,'rb') as f:
        data = pickle.load(f)
        list_of_alphabets = data['chars']
    blank_char = '_'
    list_of_alphabets.append(blank_char)
    output_character_seq = [list_of_alphabets[i] for i in input_index_seq]
    output_sentence = ''.join(output_character_seq)
    # for i in input_index_seq:
    #     output_character_seq.append(list_of_alphabets[i])

    return output_sentence

def create_mask(TIMIT_pkl_file = os.path.join(os.getcwd(),'TIMIT_data_prepared_for_CLM.pkl')):
    with open(TIMIT_pkl_file,'rb') as f:
        data = pickle.load(f)
    x = data['x']
    max_seq_len = max([len(x[i]) for i in range(len(x))])
    mask = np.zeros((len(x),max_seq_len))
    for eg_num in range(len(x)):
        mask[eg_num , 0:len(x[eg_num])] = 1
    return mask 


def prepare_TIMIT_for_CTC(dataset='train', savedir = os.getcwd(), test=0):
    print 'Getting: Inputs, Targets, Alphabet...'
    print "#########################"
    rootdir = os.path.join(TIMIT_fixed_dir,dataset)

    if (test):
        ### Read from pkl for faster testing
        in_file_name= savedir + '/TIMIT_data_prepared_for_CTC.pkl'
        with open(in_file_name, 'rb') as f:
            reclaimed_data = pickle.load(f)
            inputs = reclaimed_data['x']
            targets = reclaimed_data['y_char']
            targets_as_alphabet_indices = reclaimed_data['y_indices']
            targets_one_hot = reclaimed_data['y_onehot']
            alphabet = reclaimed_data['chars']
            sample_input = inputs[0]
            sample_target = targets[0]
            # print sample_input
            # print sample_target
    else:
        inputs,targets, alphabet= get_data(rootdir)
        print "Generating coded targets..."
        print "#########################"
        targets_as_alphabet_indices = get_TIMIT_targets_as_alphabet_indices(inputs, targets, alphabet)
        targets_one_hot = get_TIMIT_targets_one_hot(inputs, targets, alphabet)

    list_of_alphabets = [key for key in alphabet]
    list_of_alphabets.sort()
    print "Alphabet list: ", list_of_alphabets

    targets_as_alphabet_indices = [[list_of_alphabets.index(char) for char in target] for target in targets]
    print "Example target and alphabet indices: "
    print 'target = {} \nalphabet indices = {}'.format(targets[0], targets_as_alphabet_indices[0])

    # prepare file structure to store data
    n_batch = len(inputs)
    max_input_length = max([len(inputs[i]) for i in range(len(inputs))])
    input_dim = len(inputs[0][0])
    X = np.zeros((n_batch, max_input_length, input_dim))
    input_mask = np.zeros((n_batch, max_input_length))  # 1 if there's input data on this row

    # read data, store in created structures
    print "Storing data in X matrix..."
    for example_id in range(len(inputs)):
        curr_seq_len = len(inputs[example_id])
        X[example_id, :curr_seq_len] = inputs[example_id]
        input_mask[example_id, :curr_seq_len] = 1

    print  "Example of data read:"
    sample_input = inputs[0]
    sample_target = targets[0]
    print "\t input: ", sample_input
    print "\t target:", sample_target"


    ## TODO: normalize the inputs using mean.
    # From https://github.com/dtjchen/spoken-command-processor/blob/master/model/utils.py
    from sklearn import preprocessing
    def normalize_mean(X):
        scaler = preprocessing.StandardScaler(with_mean=True, with_std=False).fit(X)
        X = scaler.transform(X)
        return X, scaler.mean

    print "Normalizing input data using mean..."
    X, mean = normalize_mean(X)
    print "Mean of input data:", mean
    print "After Normalization: example of data read:"
    sample_input = inputs[0]
    sample_target = targets[0]
    print "\t input: ", sample_input
    print "\t target: sample_target"


    if (not test):
        out_file_name = savedir + '/TIMIT_data_prepared_for_CTC.pkl'
        print "Dumping to pickle file", out_file_name
        with open(out_file_name, 'wb') as f:
            pickle.dump({'x':X,
                'inputs': inputs,
                'mask': input_mask.astype(theano.config.floatX),
                'y_indices': targets_as_alphabet_indices,
                'y_char': targets,
                'y_onehot': targets_one_hot,
                'chars': list_of_alphabets}, f, protocol=2)

    print 'success!'

def prepare_TIMIT_for_CLM(dataset='train', savedir = os.getcwd(), test = 0):
    rootdir = os.path.join(TIMIT_fixed_dir, dataset)

    if (test):
        ### Read from pkl for faster testing
        in_file_name = savedir + '/TIMIT_data_prepared_for_CTC.pkl'
        with open(in_file_name, 'rb') as f:
            reclaimed_data = pickle.load(f)
            inputs = reclaimed_data['x']
            targets = reclaimed_data['y_char']
            targets_as_alphabet_indices = reclaimed_data['y_indices']
            targets_one_hot = reclaimed_data['y_onehot']
            alphabet = reclaimed_data['chars']
            sample_input = inputs[0]
            sample_target = targets[0]
            # print sample_input
            # print sample_target
    else:
        inputs, targets, alphabet = get_data(rootdir)

    t = get_TIMIT_targets_one_hot(inputs, targets, alphabet)
    t1 = get_TIMIT_targets_as_alphabet_indices(inputs, targets, alphabet)
    n_batch = len(t)
    max_input_length = max([len(t[i]) for i in range(len(t))]) - 1 #As we predict from one less than the total sequence length
    input_dim = len(t[0][0])
    X = np.zeros((n_batch, max_input_length, input_dim))
    Y = np.zeros((n_batch, max_input_length))
    input_mask = np.zeros((n_batch, max_input_length))

    for example_id in range(len(t)):
        curr_seq_len = len(t[example_id][:-1])
        X[example_id, :curr_seq_len] = t[example_id][:-1]
        input_mask[example_id, :curr_seq_len] = 1
        Y[example_id, :curr_seq_len] = t1[example_id][1:]

    # inputs = X[:,:-1,:]
    # outputs = Y[:,1:]
    inputs1 = []
    outputs1 = [
]
    for example_id in range(len(t)):
    #     # example_inputs = t[example_id][:-1]
    #     # example_outputs = t[example_id][1:]
    #     # inputs.append(example_inputs)
    #     # outputs.append(example_outputs)

        example_inputs1 = t1[example_id][:-1]
        example_outputs1 = t1[example_id][1:]
        inputs1.append(example_inputs1)
        outputs1.append(example_outputs1)

    if (not test):
        out_file_name = savedir + '/TIMIT_data_prepared_for_CLM.pkl'
        with open(out_file_name, 'wb') as f:
            # pickle.dump({'x':inputs, 'x_indices':inputs1, 'y': outputs, 'y_indices':outputs1}, f, protocol=3)
            # pickle.dump({'x':inputs.astype(theano.config.floatX), 'mask':input_mask.astype(theano.config.floatX), 'x_indices':inputs1, 'y': outputs, 'y_indices':outputs1}, f, protocol=3)
            pickle.dump({'x':X.astype(theano.config.floatX), 'mask':input_mask.astype(theano.config.floatX), 'y': Y.astype(np.int32), 'x_list': inputs1, 'y_list': outputs1}, f, protocol=2)
    # inputs = [ [ [ t[example][char] ] for char in range(0, len(t[example])-1)] for example in range(len(t))]
    # outputs = [ [ [ t[example][char] ] for char in range(1, len(t[example]))] for example in range(len(t))]
    # return inputs, outputs#, inputs1, outputs1

if __name__=='__main__':
    if len(sys.argv) > 1:
        dataset = str(sys.argv[1])
    else:
        dataset = ''
    savedir = os.getcwd()
    #pdb.set_trace()

    from fixWavs import *
    fixWavs(TIMIT_original_dir, TIMIT_fixed_dir)

    # now we still need to copy the other files (txt, phn, wrd) to the fixed dir.


    prepare_TIMIT_for_CTC(dataset, savedir, test=0)

    print("\n\n##############################")
    print("####  Preparing for CLM... ###")
    print("##############################")

    prepare_TIMIT_for_CLM(dataset, savedir, test=1)
