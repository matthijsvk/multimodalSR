#!/usr/bin/env python3

import struct
import sys
import math
import io
import os

def oracle_lna(phones_file, label_file, lna_file):
    phones = [l.strip() for l in io.open(phones_file, encoding='utf-8')]

    yes_prob = math.log(0.95)
    no_prob = math.log(0.05 / (len(phones) - 1))

    with open(lna_file, 'wb') as f:
        f.write(struct.pack(">I", len(phones)))
        f.write(struct.pack("B", 2))
        for label in io.open(label_file, encoding='utf-8'):
            i = phones.index(label.strip())
            for j in range(len(phones)):
                if i == j:
                    f.write(struct.pack(">H", int(-1820.0 * yes_prob + .5)))
                else:
                    f.write(struct.pack("BB", 255,255))

if __name__ == "__main__":
    # If we would be writing real programs, we would be doing real error handling
    phones_file = sys.argv[1]
    label_dir = sys.argv[2]
    lna_dir   = sys.argv[3]
    for f in os.listdir(label_dir):
        basename, ext = os.path.splitext(f)
	if ext != '.labels':
	   continue
	label_file = os.path.join(label_dir, f)
        lna_file = os.path.join(lna_dir, basename + '.lna')
   	oracle_lna(phones_file, label_file, lna_file)
