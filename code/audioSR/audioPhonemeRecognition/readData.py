import os
import numpy as np
from phoneme_set import *
import general_tools
import pdb

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

pickleFile = '/home/matthijs/TCDTIMIT/audioSR/TCDTIMITaudio_resampled/evaluations/volunteers_10M_predictions.pkl'
#pickleFile = '/home/matthijs/TCDTIMIT/TIMIT/binary_default/speech2phonemes26Mels/std_preprocess_26_ch.pkl'
# '/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master_2/Thesis/convNets/code/audioSR/KGP-ASR/TIMIT_Alphabet.pkl'
#  '/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master_2/Thesis/convNets/code/audioSR/KGP-ASR/TIMIT_data_prepared_for_CTC.pkl'
a = unpickle(os.path.expanduser(pickleFile))
for i in range(len(a)):
    lst = np.array(a[i])
    a[i] = lst
    print(lst.shape)

print(type(a[0]))
#[X_train, y_train, X_val, y_val, X_test, y_test] = a
[inputs, predictions, targets, avg_Acc] = a


print(inputs[0])
print(predictions[0][0])
print(targets[0])
#print(predictions[0][0] - targets[0])
print(avg_Acc)


t1 = convertNbToPhonemeList(targets[0])[1]
p1 = convertNbToPhonemeList(predictions[0])[1]
print(len(t1), len(p1))
try:assert len(t1) == len(p1)
except:pdb.set_trace()

for i in range(len(t1)):
    print(t1[i], " ", p1[i])

import pdb;pdb.set_trace()
# print(X_train.shape)
# print(X_train[0].shape)
# print(len(X_train))
# print(y_train.shape)
# print(y_train[0].shape)


#print(unpickle(os.path.expanduser(pickleFile)).keys())
