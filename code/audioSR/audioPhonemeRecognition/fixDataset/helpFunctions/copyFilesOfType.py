from __future__ import print_function

import os
import os.path
import re
import shutil
import sys

protocolPattern = re.compile('r&apos^\w+://&apos')


def pathsplit(path):
    """ This version, in contrast to the original version, permits trailing
    slashes in the pathname (in the event that it is a directory).
    It also uses no recursion """
    return path.split(os.path.sep)


def commonpath(l1, l2, common=[]):
    if len(l1) < 1: return (common, l1, l2)
    if len(l2) < 1: return (common, l1, l2)
    if l1[0] != l2[0]: return (common, l1, l2)
    return commonpath(l1[1:], l2[1:], common + [l1[0]])


def relpath(p1, p2):
    (common, l1, l2) = commonpath(pathsplit(p1), pathsplit(p2))
    p = []
    if len(l1) > 0:
        p = ['../' * len(l1)]
    p = p + l2
    return os.path.join(*p)


def isabs(string):
    if protocolPattern.match(string): return 1
    return os.path.isabs(string)


def rel2abs(path, base=os.curdir):
    if isabs(path): return path
    retval = os.path.join(base, path)
    return os.path.abspath(retval)


def abs2rel(path, base=os.curdir):  # return a relative path from base to path.
    if protocolPattern.match(path): return path
    base = rel2abs(base)
    path = rel2abs(path)  # redundant - should already be absolute
    return relpath(base, path)


def test(p1, p2):
    print("from", p1, "to", p2, " -> ",
          relpath(p1, p2))  # this is what I need. p1 = AbsDirPath; p2 = AbsfilePath; out = filepathRelToDir
    print("from", p1, "to", p2, " -> ", rel2abs(p1, p2))
    print("from", p1, "to", p2, " -> ", abs2rel(p1, p2))


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {
        "yes": True, "y": True, "ye": True,
        "no":  False, "n": False
    }
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def copyFilesOfType(srcDir, destDir, extension, interactive=False):
    print("Source Dir: %s, Destination Dir: %s, Extension: %s" % (srcDir, dstDir, extension))

    src = []
    dest = []
    for root, dirs, files in os.walk(srcDir):
        for file_ in files:
            #print(file_)
            if file_.lower().endswith(extension):
                srcPath = os.path.join(root, file_)
                relSrcPath = relpath(srcDir, srcPath).lstrip("../")
                # print(relSrcPath)
                destPath = os.path.join(dstDir, relSrcPath)
                print("copying from : %s to \t\t %s" % (srcPath, destPath))
                src.append(srcPath)
                dest.append(destPath)

    print("Example: copying ", src[0], "to:", dest[0])
    print(len(src), " files will be copied in total")

    if (interactive and (not query_yes_no("Are you sure you want to peform these operations?", "yes"))):
        print("Not doing a thing.")
    else:
        for i in range(len(src)):
            if (not os.path.exists(os.path.dirname(dest[i]))):
                os.makedirs(os.path.dirname(dest[i]))
            shutil.copy(src[i], dest[i])
        print("Done.")

    return 0


if __name__ == '__main__':
    if __name__ == '__main__':
        srcDir = sys.argv[1]
        dstDir = sys.argv[2]
        type = sys.argv[3]
        copyFilesOfType(srcDir, dstDir, type, interactive=True)

# example usage: python copyFilesOfType.py ~/Documents/Dropbox/_MyDocs/_ku_leuven/Master_2/Thesis/convNets/code /media/matthijs/TOSHIBA_EXT/TCDTIMIT/zzzNPZmodels ".npz"
