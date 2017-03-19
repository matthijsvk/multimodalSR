import sys
import os

def writeToTxt(lines, path):
    if (not os.path.exists(os.path.dirname(path))):
        os.makedirs(os.path.dirname(path))
    file = open(path, 'w')
    for line in lines:
        if (lines.index(line) < len(lines) - 1):
            file.write("%s\n" % line)
        else:
            file.write("%s" % line)
    file.close()