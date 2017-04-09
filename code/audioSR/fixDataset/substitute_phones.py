import os
import sys

# source: https://github.com/syhw/timit_tools/blob/master/src/substitute_phones.py

# example usage: python substitute_phones.py /home/matthijs/TCDTIMIT/TIMIT/temp/dr1/

"""
License: WTFPL http://www.wtfpl.net
Copyright: Gabriel Synnaeve 2013
"""

doc = """
Usage:
    python substitute_phones.py folder_path [--sentences] [--startendsil] [foldings.json]

Substitutes phones found in .lab files (in-place) by using the foldings dict.

The optional --sentences argument will replace starting and ending pauses
respectively by !ENTER and !EXIT.

The optional --startend_sil argument will replace starting and ending phones by 'sil'
"""

from phoneme_set import phoneme_set_61_39


def process(folder,
            sentences=False,  # should we apply !ENTER/!EXIT for start/end?
            startend_sil=False):  # should we substitute start and end w/ sil
    foldings = phoneme_set_61_39
    c_before = {}
    c_after = {}
    for d, ds, fs in os.walk(folder):
        for fname in fs:
            if not (fname[-4:] == '.phn' or fname[-4:] == '.PHN'):
                continue
            fullname = d.rstrip('/') + '/' + fname
            phones_before = []
            phones_after = []
            os.rename(fullname, fullname + '~')
            fr = open(fullname + '~', 'r')
            fw = open(fullname, 'w')
            text_buffer = []
            for line in fr:
                phones_before.append(line.split()[-1])  # phone last elt of line
                tmpline = line
                tmpline = tmpline.replace('-', '')
                tmp = tmpline.split()
                for k, v in foldings.iteritems():
                    if tmp[-1] == k:
                        tmp[-1] = v
                        tmpline = ' '.join(tmp)
                text_buffer.append(tmpline.split())
            first_phone = text_buffer[0][-1].strip()
            last_phone = text_buffer[-1][-1].strip()

            if sentences:
                if first_phone == 'h#' or first_phone == 'sil' or first_phone == '<s>' or first_phone == '{B_TRANS}':
                    # 'h#' or 'sil' for TIMIT
                    # '<s>' for CSJ (and other XML/Thomas-like)
                    # '{B_TRANS}' for Buckeye
                    text_buffer[0] = text_buffer[0][:-1] + ['!ENTER']
                if last_phone == 'h#' or last_phone == 'sil' or last_phone == '</s>' or last_phone == '{E_TRANS}':
                    text_buffer[-1] = text_buffer[-1][:-1] + ['!EXIT']

            if startend_sil:
                text_buffer[0] = text_buffer[0][:-1] + ['sil']
                text_buffer[-1] = text_buffer[-1][:-1] + ['sil']

            for buffer_line in text_buffer:
                phones_after.append(buffer_line[-1])
                fw.write(' '.join(buffer_line) + '\n')
            fw.close()
            os.remove(fullname + '~')

            for tmp_phn in phones_before:
                c_before[tmp_phn] = c_before.get(tmp_phn, 0) + 1
            for tmp_phn in phones_after:
                c_after[tmp_phn] = c_after.get(tmp_phn, 0) + 1
            print "dealt with", fullname
    print "Counts before substitution", sorted(c_before.items())
    print "Counts after substitution", sorted(c_after.items())
    print "nb phonemes before: ", len(c_before)
    print "nb phonemes after: ", len(c_after)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if '--help' in sys.argv:
            print doc
            sys.exit(0)
        foldername = sys.argv[1]
        print foldername

        sentences = False
        foldings = {}
        startend_sil = False
        if '--sentences' in sys.argv:
            sentences = True
        if '--startendsil' in sys.argv:
            startend_sil = True
        process(foldername, sentences, startend_sil)
    else:
        print doc
