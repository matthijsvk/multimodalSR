import os
import urllib
import requests

# 1. read txt file that contains of links to volunteers (or lipreaders), create folder structure and return directory of folder names and URLs
#    Each line of the txt file is as follows:
#        folderName: URL
#    eg: 01M: https://sigmedia.tcd.ie/TCDTIMIT/filebrowser/download/1223
def readfile(filename):
    with open(filename, "r") as ins:
        URLdirectory = {}
        video_index = 0
        for line in ins:
            line = line.strip('\n') # strip newlines
            if len(line)>1: # don't save the dots lines
                folderName, URLname = line.split(": ")
                URLdirectory[folderName] = URLname

    return URLdirectory # directory of folder names and URLs

# 2. read directory of folder names and URLs, download the URLs to the correct folder
def downloadFiles(baseDir, URLdirectory):
    urlList = sorted(URLdirectory.iteritems())
    i=0;
    for url in urlList:
        if i<1:
            print(url)
            # get the file
            response = requests.get(url[1], stream=True)
            print(response)
            # store it
            fname = baseDir+os.sep+url[0]+os.sep+url[0]+".zip"
            # check if file and dir structure exist, create dirs if needed
            if not os.path.exists(os.path.dirname(fname)):
                try:
                    os.makedirs(os.path.dirname(fname))
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            # actually write to the file as a stream: http://stackoverflow.com/questions/14114729/save-a-large-file-using-the-python-requests-library#14114741
            with open(fname, 'w') as handle:
                if not response.ok:
                    # Something went wrong
                    print("something went wrong...")

                for block in response.iter_content(1024):
                    handle.write(block)
        i+=1
    return 0
        
# testing

# URLdirectory = readfile("/home/matthijs/Documents/Dropbox/_MyDocs/_ku_leuven/Master/Thesis/Datasets/TCD-TIMIT/downloadLocationVolunteers.txt")
# downloadFiles("/media/toshiba/TCDTIMIT/volunteers",URLdirectory)


import httplib
c = httplib.HTTPSConnection("sigmedia.tcd.ie/TCDTIMIT")
c.request("GET", "/")
response = c.getresponse()
print response.status, response.reason
data = response.read()
print data
