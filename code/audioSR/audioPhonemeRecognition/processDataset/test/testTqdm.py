from tqdm import tqdm
from time import sleep

wav_files = ['a/b/c', 'a/b/d', 'a/b/e','b/c/e']

for wavFile in tqdm(wav_files, total=len(wav_files)):
    #print(wavFile)
    sleep(0.25)