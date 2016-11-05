# from http://stackoverflow.com/questions/10672578/extract-video-frames-in-python#10672679

# Goal: parametrized, automated version of 
#       ffmpeg -i n.mp4 -ss 00:00:20 -s 160x120 -r 1 -f singlejpeg myframe.jpg

import os, sys
from PIL import Image
from resizeimage import resizeimage
from subprocess import call


inputname= 'sa1.mp4'
time = '00:00:01.150'
size = '160x120'
outputname= "sa1.jpg" 

# from https://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/
import subprocess as sp
command = ['ffmpeg',
            '-ss', time,
            '-i', inputname,
            '-ss', '1',
	    '-f','singlejpeg',outputname]
pipe = call(command)


