
import shutil
from general_tools import *
badSpeakersDirs = unpickle('./badDirs.pkl')
badVideoDirs = unpickle('./badDirs2.pkl')

targetDir = os.path.expanduser('~/TCDTIMIT/lipreading/processed')
rootDir = '/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/database/'


deleted = []
# for badSpeaker in badSpeakersDirs:
#     for badDir in badSpeaker:
for badDir in badVideoDirs:
    # eg /home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/database/lipspeakers/Lipspkr1/sx180
    # We want to convert to
    # /home/matthijs/TCDTIMIT/lipreading/processed/lipspeakers/Lipspkr1/sx180
    relPath = relpath(rootDir, badDir)
    relTopDir = relPath.split('/')[0]
    while not (relTopDir == 'lipspeakers' or relTopDir == 'volunteers'):
        relPath = '/'.join(relPath.split('/')[1:])
        relTopDir = relPath.split('/')[0]
    print(relPath)

    toDelete = os.path.join(targetDir, relPath)
    print("TO DELETE: ", toDelete)

    try:
        #shutil.rmtree(toDelete);
        deleted.append(toDelete)
    except:
        continue

print(len(deleted))
import pdb;pdb.set_trace()