
import shutil
from general_tools import *
# badSpeakersDirs = unpickle('./badDirs.pkl')
# badVideoDirs = unpickle('./badDirs2.pkl')

badVideoDirs = ['/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/database/lipspeakers/Lipspkr2/si1585', '/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/database/lipspeakers/Lipspkr3/si1077', '/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/database/volunteers/04M/si674', '/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/database/volunteers/28M/sx337', '/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/database/volunteers/27M/si948', '/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/database/volunteers/08F/si546', '/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/database/volunteers/10M/sx121', '/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/database/volunteers/35M/si1259', '/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/database/volunteers/29M/si1424']
badVideoDirs = ['/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/database/lipspeakers/Lipspkr2/si1583', '/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/database/lipspeakers/Lipspkr2/si1592', '/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/database/volunteers/28M/sx337', '/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/database/volunteers/27M/si948', '/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/database/volunteers/08F/si546', '/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/database/volunteers/10M/sx121', '/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/database/volunteers/35M/si1259', '/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/database/volunteers/29M/si1424']
badVideoDirs = ['/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/database/volunteers/28M/sx337', '/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/database/volunteers/27M/si948', '/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/database/volunteers/08F/si546', '/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/database/volunteers/10M/sx121', '/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/database/volunteers/35M/si1259', '/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/database/volunteers/29M/si1424']

processedDir = os.path.expanduser('~/TCDTIMIT/lipreading/processed')

databaseDir = '/home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/database/'

deleted = []
# for badSpeaker in badSpeakersDirs:
#     for badDir in badSpeaker:
for badDir in badVideoDirs:
    # eg /home/matthijs/TCDTIMIT/combinedSR/TCDTIMIT/database/lipspeakers/Lipspkr1/sx180
    # We want to convert to
    # /home/matthijs/TCDTIMIT/lipreading/processed/lipspeakers/Lipspkr1/sx180
    relPath = relpath(databaseDir, badDir)
    relTopDir = relPath.split('/')[0]
    while not (relTopDir == 'lipspeakers' or relTopDir == 'volunteers'):
        relPath = '/'.join(relPath.split('/')[1:])
        relTopDir = relPath.split('/')[0]
    print(relPath)

    toDeleteDatabase = os.path.join(databaseDir, relPath)
    toDeleteProcessed = os.path.join(processedDir, relPath)
    print("TO DELETE: ", toDeleteProcessed)

    try:
        shutil.rmtree(toDeleteDatabase); shutil.rmtree(toDeleteProcessed)
        deleted.append(toDeleteDatabase); deleted.append(toDeleteProcessed)
    except:
        continue

print(len(deleted))
import pdb;pdb.set_trace()

