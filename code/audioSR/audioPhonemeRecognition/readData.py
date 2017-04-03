import os
import numpy as np

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

pickleFile = '/home/matthijs/TCDTIMIT/TIMIT/binary_default/speech2phonemes26Mels/std_preprocess_26_ch.pkl'
# '/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master_2/Thesis/convNets/code/audioSR/KGP-ASR/TIMIT_Alphabet.pkl'
#  '/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master_2/Thesis/convNets/code/audioSR/KGP-ASR/TIMIT_data_prepared_for_CTC.pkl'
a = unpickle(os.path.expanduser(pickleFile))
for i in range(len(a)):
    lst = np.array(a[i])
    a[i] = lst
    print(lst.shape)

print(type(a[0]))
[X_train, y_train, X_val, y_val, X_test, y_test] = a

print(X_train.shape)
print(X_train[0].shape)
print(len(X_train))
print(y_train.shape)
print(y_train[0].shape)


#print(unpickle(os.path.expanduser(pickleFile)).keys())
