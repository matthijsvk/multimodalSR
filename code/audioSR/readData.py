import os
import sys

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

print(unpickle(os.path.expanduser('/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master_2/Thesis/convNets/code/audioSR/KGP-ASR/TIMIT_Alphabet.pkl')))

print(unpickle(os.path.expanduser('/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master_2/Thesis/convNets/code/audioSR/KGP-ASR/TIMIT_data_prepared_for_CTC.pkl')).keys())