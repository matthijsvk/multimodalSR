import os
import sys

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


# '/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master_2/Thesis/convNets/code/audioSR/KGP-ASR/TIMIT_Alphabet.pkl'
# '/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master_2/Thesis/convNets/code/audioSR/KGP-ASR/TIMIT_data_prepared_for_CTC.pkl'

pickleFile = os.path.expanduser("~/TCDTIMIT/TIMIT/fixedWav/TIMIT/std_preprocess_26_ch_DEBUG.pkl")
a= unpickle(pickleFile)

print(len(a))
print("\n")
for i in range(len(a)):
    print(len(a[i]))
    print(len(a[i][0]))
    try: print(len(a[i][0][0]))
    except: pass

    print(a[i][0][0])

    print("\n")


print(a.keys())
