import os
import sys

from utils import helpFunctions as wrTxt

"""
License: WTFPL http://www.wtfpl.net
Copyright: Gabriel Synnaeve 2013
"""

doc = """
Usage:
    python createMLF.py [$folder_path]

Will create "${folder}.mlf" and "labels.txt"  files in $folder_path.

If you run it only on the training folder, all the phones that you will
encounter in the test should be present in training so that the "labels"
corresponds.
"""

def process(folder):
    folder = folder.rstrip('/')
    countPhonemes = {}
    master_label_fname = folder + '/' + folder.split('/')[-1] + '.mlf'
    labels_fpath = folder + '/labels.txt'
    master_label_file = open(master_label_fname, 'w')
    master_label_file.write("#!MLF!#\n")

    for d, ds, fs in os.walk(folder):
        for fname in fs:
            fullname = d.rstrip('/') + '/' + fname
            print("Processing: ", fullname)
            extension = fname[-4:]

            phones = []
            if extension.lower() == '.phn':
                master_label_file.write('"' + fullname + '"\n')
                for line in open(fullname):
                    master_label_file.write(line)
                    phones.append(line.split()[2])
                for tmp_phn in phones:
                    countPhonemes[tmp_phn] = countPhonemes.get(tmp_phn, 0) + 1
                master_label_file.write('\n.\n')

    master_label_file.close()
    print("written MLF file", master_label_fname)

    wrTxt.writeToTxt(sorted(countPhonemes.items()), labels_fpath)
    print("written labels", labels_fpath)

    print("phones counts:", countPhonemes)
    print("number of phones:", len(countPhonemes))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if '--help' in sys.argv:
            print(doc)
            sys.exit(0)
        l = filter(lambda x: not '--' in x[0:2], sys.argv)
        print(l)
        foldername = '.'
        if len(l) > 1:
            foldername = l[1]
            print(foldername)
        process(foldername)
    else:
        process('.') # default