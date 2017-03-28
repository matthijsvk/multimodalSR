import os
from shutil import copyfile
import subprocess

def transform_wav(wav_file, output):
    filename = os.path.basename(wav_file)
    pathname = os.path.dirname(wav_file)
    speaker = os.path.basename(pathname)
    dr = os.path.basename(os.path.dirname(pathname))
    output_dir = os.path.join(output, dr, speaker)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fixedWavPath = os.path.join(output_dir, filename)

    if not os.path.exists(fixedWavPath):
        command = ['mplayer',
                   '-quiet',
                   '-vo', 'null',
                   '-vc', 'dummy',
                   '-ao', 'pcm:waveheader:file='+fixedWavPath,
                   wav_file]

        # actually run the command, only show stderror on terminal, close the processes (don't wait for user input)
        FNULL = open(os.devnull, 'w')
        p = subprocess.Popen(command, stdout=FNULL, stderr=subprocess.STDOUT, close_fds=True)  # stdout=subprocess.PIPE
        return 1
    else:
        return 0


def copy_phn(phn_file, output):
    filename = os.path.basename(phn_file)
    pathname = os.path.dirname(phn_file)
    speaker = os.path.basename(pathname)
    dr = os.path.basename(os.path.dirname(pathname))
    output_file = os.path.join(output, dr, speaker, filename)

    copyfile(phn_file, output_file)


