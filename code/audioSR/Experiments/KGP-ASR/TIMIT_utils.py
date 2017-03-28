####!!!!!!!!!!!!!!! OUTPUT VOCABULARY GENERATION REMAINS !!!!!!!!!!!!!!!!!!####
# import pdb
import numpy as np
import os
import soundfile as sf
from features import mfcc
import pickle
import sys
import theano
import pdb
#==============================================================
#TODO: NORMALIZE AND INCORPORATE DELTA AND DELTA-DELTA FEATURES
#==============================================================

# from features import logfbank

# class MFCC_input():
# 	def __init__(self,sequence = None):
# 		self.sequence = sequence
# 	def get_sequence(self):
# 		return self.sequence
# 	def get_sequence_length(self):
# 		return self.sequence.shape[0]

# class character_output():
# 	def __init__(self,sequence = None):
# 		self.sequence = sequence
# 	def get_sequence(self):
# 		return self.sequence
# 	def getget_sequence_length(self):
# 		return len(self.sequence)

TIMIT_main_dir = '/home/anirban/Desktop/TimeForAnUpgrade/DeepLearningWithPython/TIMITDataPrepared/TIMIT/timit'

def get_alphabet(rootdir = TIMIT_main_dir):
	#This is a dictionary with key-value pairs where the keys are charcters and values are frequencies
	alphabet = {} 
	for pathname, subdirectories, filenames in os.walk(rootdir):
		for filename in filenames:
			if (os.path.join(pathname,filename)).endswith('.wav'):
				transcription_filename = os.path.join(pathname,filename)[:-4] + '.txt'
				transcription_file = open(transcription_filename,'r')
				transcription = str(transcription_file.read()).lower().translate(None, '!:,".;?')
				# transcription = str(transcription_file.read()).lower().translate(str.maketrans('','', '!:,".;?'))

				transcription = transcription[8:-1]
				for char in transcription:
					if not char in alphabet:
						alphabet.update({char:1})
					else:
						alphabet[char] += 1
	
	# # print 'TIMIT Alphabet:\n', alphabet
	# alphabet_filename = 'TIMIT_Alphabet.pkl'
	# with open(alphabet_filename,'wb') as f:
	# 	pickle.dump(alphabet,f,protocol=3)
	return alphabet 

def get_data(rootdir = TIMIT_main_dir):	
	inputs = []
	targets = []
	for dir_path, sub_dirs, files in os.walk(rootdir):
		for file in files:	        
			if (os.path.join(dir_path, file)).endswith('.wav'):
				wav_file_name = os.path.join(dir_path, file)
				input_data, f_s = sf.read(wav_file_name)
				# mfcc_feat = MFCC_input(mfcc(input_data,f_s))
				mfcc_feat = mfcc(input_data,f_s)
				#Delta features
				delta_feat = mfcc_feat[:-1]-mfcc_feat[1:]
				#Delta-Delta features
				deltadelta_feat = delta_feat[:-1]-delta_feat[1:]

				#Removing the first two frames
				mfcc_feat = mfcc_feat[2:]
				delta_feat = delta_feat[1:]

				#Concatenating mfcc, delta and delta-delta features
				full_input = np.concatenate((mfcc_feat,delta_feat,deltadelta_feat), axis=1)

				inputs.append(np.asarray(full_input, dtype=theano.config.floatX))#Rakeshvar wants one frame along each column but i am using Lasagne

				text_file_name = wav_file_name[:-4] + '.txt'
				target_data_file = open(text_file_name)
				target_data = str(target_data_file.read()).lower().translate(None, '!:,".;?')
				# target_data = str(target_data_file.read()).lower().translate(str.maketrans('','', '!:,".;?'))
				target_data = target_data[8:-1]#No '.' in lexfree dictionary
				targets.append(target_data)
	return inputs, targets



def get_TIMIT_targets_one_hot(rootdir = TIMIT_main_dir):
	alphabet = get_alphabet(rootdir);
	list_of_alphabets = [key for key in alphabet]
	list_of_alphabets.sort()
	# print list_of_alphabets
	num_targets = len(list_of_alphabets)
	inputs,targets = get_data(rootdir)
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

def get_TIMIT_targets_as_alphabet_indices(rootdir = TIMIT_main_dir):
	alphabet = get_alphabet(rootdir);
	list_of_alphabets = [key for key in alphabet]
	list_of_alphabets.sort()
	# print('list of alphabets: {}'.format(list_of_alphabets))
	inputs,targets = get_data(rootdir)
	targets_as_alphabet_indices = [[list_of_alphabets.index(char) for char in example] for example in targets]
	# print('target = {} \n alphabet indices = {}'.format(targets[0], targets_as_alphabet_indices[0]))

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
	# 	output_character_seq.append(list_of_alphabets[i])

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


def prepare_TIMIT_for_CTC(dataset='train', savedir = os.getcwd()):
	rootdir = os.path.join(TIMIT_main_dir,dataset)
	inputs,targets = get_data(rootdir)
	alphabet = get_alphabet(rootdir)
	targets_as_alphabet_indices = get_TIMIT_targets_as_alphabet_indices(rootdir)
	targets_one_hot = get_TIMIT_targets_one_hot(rootdir)
	list_of_alphabets = [key for key in alphabet]
	list_of_alphabets.sort()

	n_batch = len(inputs)
	max_input_length = max([len(inputs[i]) for i in range(len(inputs))])
	input_dim = len(inputs[0][0])
	X = np.zeros((n_batch, max_input_length, input_dim))
	input_mask = np.zeros((n_batch, max_input_length))

	for example_id in range(len(inputs)):
		curr_seq_len = len(inputs[example_id])
		X[example_id, :curr_seq_len] = inputs[example_id]
		input_mask[example_id, :curr_seq_len] = 1

	# sample_input = inputs[0]
	# sample_target = targets[0]
	# print sample_input
	# print sample_target
	out_file_name = savedir + '/TIMIT_data_prepared_for_CTC.pkl'
	with open(out_file_name, 'wb') as f:
		# pickle.dump({'x':inputs, 'y_indices': targets_as_alphabet_indices, 'y_char': targets, 'y_onehot': targets_one_hot, 'chars': list_of_alphabets}, f, protocol=3)
		pickle.dump({'x':X,
			'inputs': inputs, 
			'mask': input_mask.astype(theano.config.floatX), \
			'y_indices': targets_as_alphabet_indices, 
			'y_char': targets, 'y_onehot': targets_one_hot, 
			'chars': list_of_alphabets}, f, protocol=2)

	# with open(out_file_name, 'rb') as f:
	# 	reclaimed_data = pickle.load(f)
	# 	inputs = reclaimed_data['x']
	# 	targets = reclaimed_data['y_indices']
	# 	sample_input = inputs[0]
	# 	sample_target = targets[0]
	# 	# print sample_input
	# 	# print sample_target

	# print 'success!'

def prepare_TIMIT_for_CLM(dataset='train', savedir = os.getcwd()):
	rootdir = os.path.join(TIMIT_main_dir, dataset)
	t = get_TIMIT_targets_one_hot(rootdir)
	t1 = get_TIMIT_targets_as_alphabet_indices(rootdir)
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
	# 	# example_inputs = t[example_id][:-1]
	# 	# example_outputs = t[example_id][1:]
	# 	# inputs.append(example_inputs)
	# 	# outputs.append(example_outputs)

		example_inputs1 = t1[example_id][:-1]
		example_outputs1 = t1[example_id][1:]
		inputs1.append(example_inputs1)
		outputs1.append(example_outputs1)

	out_file_name = savedir + '/TIMIT_data_prepared_for_CLM.pkl'
	with open(out_file_name, 'wb') as f:
		# pickle.dump({'x':inputs, 'x_indices':inputs1, 'y': outputs, 'y_indices':outputs1}, f, protocol=3)
		# pickle.dump({'x':inputs.astype(theano.config.floatX), 'mask':input_mask.astype(theano.config.floatX), 'x_indices':inputs1, 'y': outputs, 'y_indices':outputs1}, f, protocol=3)
		pickle.dump({'x':X.astype(theano.config.floatX), 'mask':input_mask.astype(theano.config.floatX), 'y': Y.astype(np.int32), 'x_list': inputs1, 'y_list': outputs1}, f, protocol=2)
	# inputs = [ [ [ t[example][char] ] for char in range(0, len(t[example])-1)] for example in range(len(t))]
	# outputs = [ [ [ t[example][char] ] for char in range(1, len(t[example]))] for example in range(len(t))]
	# return inputs, outputs#, inputs1, outputs1

# def prepare_TIMIT_for_CLM_temp(dataset='train', savedir = os.getcwd()):
# 	rootdir = TIMIT_main_dir + '/' + dataset
# 	t = get_TIMIT_targets_as_alphabet_indices(rootdir)
# 	inputs = []
# 	outputs = []
# 	for example_id in range(len(t)):
# 		example_inputs = t[example_id][:-1]
# 		example_outputs = t[example_id][1:]
# 		inputs.append(example_inputs)
# 		outputs.append(example_outputs)
# 	# inputs = [ [ [ t[example][char] ] for char in range(0, len(t[example])-1)] for example in range(len(t))]
# 	# outputs = [ [ [ t[example][char] ] for char in range(1, len(t[example]))] for example in range(len(t))]
# 	return inputs, outputs

if __name__=='__main__':
	if len(sys.argv) > 1:
		dataset = str(sys.argv[1])
	else:
		dataset = ''
	savedir = os.getcwd()
	pdb.set_trace()
	prepare_TIMIT_for_CTC(dataset, savedir)
	prepare_TIMIT_for_CLM(dataset, savedir)
